using System.Globalization;
using System.Net.Http;
using System.Text.Json;

CliOptions options = CliOptions.Parse(args);

if (options.ShowHelp)
{
    PrintHelp(options.Error);
    return options.Error is null ? 0 : 2;
}

IdsDoctor doctor = new(options);
return await doctor.RunAsync();

static void PrintHelp(string? error)
{
    if (!string.IsNullOrWhiteSpace(error))
    {
        Console.WriteLine($"Error: {error}");
        Console.WriteLine();
    }

    Console.WriteLine("IDS Doctor - optional read-only diagnostics for the IDS project");
    Console.WriteLine();
    Console.WriteLine("Usage:");
    Console.WriteLine("  dotnet run --project tools/IdsDoctor");
    Console.WriteLine("  dotnet run --project tools/IdsDoctor -- --root C:\\project\\ids");
    Console.WriteLine();
    Console.WriteLine("Options:");
    Console.WriteLine("  --root <path>        Project root. Defaults to auto-detect from current directory.");
    Console.WriteLine("  --health-url <url>   Health endpoint to check. Defaults to config host/port + /health.");
    Console.WriteLine("  --skip-health        Skip the optional HTTP health check.");
    Console.WriteLine("  --help              Show this help.");
}

internal sealed class IdsDoctor
{
    private static readonly string[] RequiredArtifactNames =
    [
        "binary_model_weights_best.npz",
        "multiclass_model_weights_best.npz",
        "binary_preprocess.npz",
        "multiclass_preprocess.npz",
        "multiclass_label_map.json"
    ];

    private readonly CliOptions _options;
    private readonly DiagnosticReport _report = new();

    public IdsDoctor(CliOptions options)
    {
        _options = options;
    }

    public async Task<int> RunAsync()
    {
        string root = ResolveProjectRoot(_options.Root);
        string configPath = Path.Combine(root, "config.json");

        _report.Title("IDS Doctor");
        _report.Info("Mode", "Read-only diagnostics; no Python code, artifacts, or logs are modified.");
        _report.Info("Project root", root);

        JsonDocument? runtimeConfig = LoadJson(configPath, "Runtime config");
        try
        {
            string artifactsPath = ResolveArtifactsPath(root, runtimeConfig);

            CheckRuntimeConfig(runtimeConfig);
            CheckArtifacts(artifactsPath);
            CheckTrainingData(root);
            CheckBinaryReport(Path.Combine(artifactsPath, "train_eval_report.json"));
            CheckMulticlassReport(Path.Combine(artifactsPath, "multiclass_eval_report.json"));
            CheckLabelMap(Path.Combine(artifactsPath, "multiclass_label_map.json"));

            if (!_options.SkipHealth)
            {
                string healthUrl = _options.HealthUrl ?? BuildDefaultHealthUrl(runtimeConfig);
                await CheckHealthAsync(healthUrl);
            }
        }
        finally
        {
            runtimeConfig?.Dispose();
        }

        return _report.Summary();
    }

    private void CheckRuntimeConfig(JsonDocument? config)
    {
        _report.Section("Runtime config");

        if (config is null)
        {
            return;
        }

        JsonElement root = config.RootElement;
        CheckRequiredString(root, "flask_host");
        CheckPort(root, "flask_port");
        CheckRange(root, "binary_threshold", 0, 1);
        CheckRequiredString(root, "artifacts_path");
        CheckRequiredString(root, "agent_server_url");
        CheckPositiveNumber(root, "request_timeout_seconds");
        CheckPositiveNumber(root, "traffic_store_limit");
    }

    private void CheckArtifacts(string artifactsPath)
    {
        _report.Section("Runtime artifacts");

        if (!Directory.Exists(artifactsPath))
        {
            _report.Fail("Artifacts directory", $"Missing directory: {artifactsPath}");
            return;
        }

        _report.Ok("Artifacts directory", artifactsPath);

        foreach (string artifactName in RequiredArtifactNames)
        {
            string path = Path.Combine(artifactsPath, artifactName);
            CheckFileExists(path, artifactName, requireNonEmpty: true);
        }
    }

    private void CheckTrainingData(string root)
    {
        _report.Section("Training data");
        CheckFileExists(Path.Combine(root, "data", "raw", "KDDTrain+.txt"), "KDDTrain+.txt", requireNonEmpty: true);
        CheckFileExists(Path.Combine(root, "data", "raw", "KDDTest+.txt"), "KDDTest+.txt", requireNonEmpty: true);
    }

    private void CheckBinaryReport(string reportPath)
    {
        _report.Section("Binary model report");

        JsonDocument? report = LoadJson(reportPath, "Binary evaluation report");
        if (report is null)
        {
            return;
        }

        using (report)
        {
            JsonElement root = report.RootElement;
            ReportMetric(root, "test.acc", "Test accuracy");
            ReportMetric(root, "test.f1", "Test F1");
            ReportRows(root, "test.rows", "Test rows");
            ReportMetric(root, "val.acc", "Validation accuracy");
            ReportMetric(root, "val.f1", "Validation F1");

            double? testF1 = JsonHelpers.GetDouble(root, "test.f1");
            if (testF1.HasValue && testF1.Value < 0.70)
            {
                _report.Warn("Binary test F1", $"Below 0.70 ({FormatDecimal(testF1.Value)}).");
            }
            else if (testF1.HasValue)
            {
                _report.Ok("Binary test F1", $"{FormatDecimal(testF1.Value)} is above the 0.70 warning threshold.");
            }
        }
    }

    private void CheckMulticlassReport(string reportPath)
    {
        _report.Section("Multiclass model report");

        JsonDocument? report = LoadJson(reportPath, "Multiclass evaluation report");
        if (report is null)
        {
            return;
        }

        using (report)
        {
            JsonElement root = report.RootElement;
            ReportMetric(root, "test.acc", "Test accuracy");
            ReportMetric(root, "test.macro_f1", "Test macro F1");
            ReportMetric(root, "test.macro_precision", "Test macro precision");
            ReportMetric(root, "test.macro_recall", "Test macro recall");
            ReportRows(root, "test.rows", "Test rows");

            double? macroF1 = JsonHelpers.GetDouble(root, "test.macro_f1");
            if (macroF1.HasValue && macroF1.Value < 0.70)
            {
                _report.Warn("Multiclass macro F1", $"Below 0.70 ({FormatDecimal(macroF1.Value)}). Check minority classes.");
            }
            else if (macroF1.HasValue)
            {
                _report.Ok("Multiclass macro F1", $"{FormatDecimal(macroF1.Value)} is above the 0.70 warning threshold.");
            }

            if (!JsonHelpers.TryGetElement(root, "test.per_class", out JsonElement perClass) ||
                perClass.ValueKind != JsonValueKind.Object)
            {
                _report.Warn("Per-class metrics", "Missing test.per_class object.");
                return;
            }

            foreach (JsonProperty classMetrics in perClass.EnumerateObject())
            {
                string className = classMetrics.Name;
                JsonElement metrics = classMetrics.Value;
                double? f1 = JsonHelpers.GetDouble(metrics, "f1");
                int? support = JsonHelpers.GetInt(metrics, "support");

                if (!f1.HasValue)
                {
                    _report.Warn($"{className} F1", "Missing f1 metric.");
                    continue;
                }

                string supportText = support.HasValue ? $", support {support.Value}" : string.Empty;
                if (f1.Value < 0.20)
                {
                    _report.Warn($"{className} F1", $"{FormatDecimal(f1.Value)}{supportText}. This class needs attention.");
                }
                else
                {
                    _report.Ok($"{className} F1", $"{FormatDecimal(f1.Value)}{supportText}.");
                }

                if (support.HasValue && support.Value > 0 && support.Value < 100)
                {
                    _report.Warn($"{className} support", $"Only {support.Value} test rows. Metrics may be unstable.");
                }
            }
        }
    }

    private void CheckLabelMap(string path)
    {
        _report.Section("Attack label map");

        JsonDocument? labelMap = LoadJson(path, "Multiclass label map");
        if (labelMap is null)
        {
            return;
        }

        using (labelMap)
        {
            JsonElement root = labelMap.RootElement;

            if (JsonHelpers.TryGetElement(root, "family_names", out JsonElement familyNames) &&
                familyNames.ValueKind == JsonValueKind.Array)
            {
                _report.Ok("Attack families", $"{familyNames.GetArrayLength()} families: {string.Join(", ", familyNames.EnumerateArray().Select(ToDisplayString))}.");
            }
            else
            {
                _report.Warn("Attack families", "Missing family_names array.");
            }

            if (JsonHelpers.TryGetElement(root, "attack_family_map", out JsonElement attackMap) &&
                attackMap.ValueKind == JsonValueKind.Object)
            {
                _report.Ok("Attack mappings", $"{attackMap.EnumerateObject().Count()} attack labels mapped to families.");
            }
            else
            {
                _report.Warn("Attack mappings", "Missing attack_family_map object.");
            }
        }
    }

    private async Task CheckHealthAsync(string healthUrl)
    {
        _report.Section("Optional server health");

        using HttpClient client = new()
        {
            Timeout = TimeSpan.FromMilliseconds(1500)
        };

        try
        {
            using HttpResponseMessage response = await client.GetAsync(healthUrl);
            string responseBody = await response.Content.ReadAsStringAsync();

            if (!response.IsSuccessStatusCode)
            {
                _report.Warn("Health endpoint", $"{healthUrl} returned HTTP {(int)response.StatusCode}.");
                return;
            }

            using JsonDocument health = JsonDocument.Parse(responseBody);
            JsonElement root = health.RootElement;
            string status = JsonHelpers.GetString(root, "status") ?? "unknown";
            bool? modelsLoaded = JsonHelpers.GetBool(root, "models_loaded");
            int? trafficCount = JsonHelpers.GetInt(root, "traffic_count");
            int? attackCount = JsonHelpers.GetInt(root, "attack_count");
            bool? agentRunning = JsonHelpers.GetBool(root, "agent_running");

            _report.Ok("Health endpoint", $"{healthUrl} returned status '{status}'.");

            if (modelsLoaded == true)
            {
                _report.Ok("Runtime models", "Server reports models_loaded=true.");
            }
            else
            {
                _report.Warn("Runtime models", "Server did not report models_loaded=true.");
            }

            _report.Info("Runtime counters", $"traffic_count={trafficCount?.ToString(CultureInfo.InvariantCulture) ?? "unknown"}, attack_count={attackCount?.ToString(CultureInfo.InvariantCulture) ?? "unknown"}.");
            _report.Info("Agent state", $"agent_running={agentRunning?.ToString() ?? "unknown"}.");
        }
        catch (Exception ex) when (ex is HttpRequestException or TaskCanceledException or JsonException)
        {
            _report.Warn("Health endpoint", $"{healthUrl} is not reachable or did not return valid JSON ({ex.GetType().Name}).");
        }
    }

    private JsonDocument? LoadJson(string path, string label)
    {
        if (!File.Exists(path))
        {
            _report.Fail(label, $"Missing file: {path}");
            return null;
        }

        try
        {
            JsonDocument document = JsonDocument.Parse(File.ReadAllText(path));
            _report.Ok(label, $"Loaded {path}.");
            return document;
        }
        catch (JsonException ex)
        {
            _report.Fail(label, $"Invalid JSON in {path}: {ex.Message}");
            return null;
        }
    }

    private void CheckRequiredString(JsonElement root, string path)
    {
        string? value = JsonHelpers.GetString(root, path);
        if (string.IsNullOrWhiteSpace(value))
        {
            _report.Fail(path, "Missing or empty string.");
            return;
        }

        _report.Ok(path, value);
    }

    private void CheckPort(JsonElement root, string path)
    {
        int? value = JsonHelpers.GetInt(root, path);
        if (!value.HasValue || value.Value < 1 || value.Value > 65535)
        {
            _report.Fail(path, "Expected a TCP port between 1 and 65535.");
            return;
        }

        _report.Ok(path, value.Value.ToString(CultureInfo.InvariantCulture));
    }

    private void CheckRange(JsonElement root, string path, double min, double max)
    {
        double? value = JsonHelpers.GetDouble(root, path);
        if (!value.HasValue || value.Value < min || value.Value > max)
        {
            _report.Fail(path, $"Expected a number between {FormatDecimal(min)} and {FormatDecimal(max)}.");
            return;
        }

        _report.Ok(path, FormatDecimal(value.Value));
    }

    private void CheckPositiveNumber(JsonElement root, string path)
    {
        double? value = JsonHelpers.GetDouble(root, path);
        if (!value.HasValue || value.Value <= 0)
        {
            _report.Fail(path, "Expected a positive number.");
            return;
        }

        _report.Ok(path, FormatDecimal(value.Value));
    }

    private void CheckFileExists(string path, string label, bool requireNonEmpty)
    {
        if (!File.Exists(path))
        {
            _report.Fail(label, $"Missing file: {path}");
            return;
        }

        FileInfo file = new(path);
        if (requireNonEmpty && file.Length == 0)
        {
            _report.Fail(label, $"File is empty: {path}");
            return;
        }

        _report.Ok(label, $"{path} ({FormatBytes(file.Length)}).");
    }

    private void ReportMetric(JsonElement root, string path, string label)
    {
        double? value = JsonHelpers.GetDouble(root, path);
        if (!value.HasValue)
        {
            _report.Warn(label, $"Missing metric: {path}.");
            return;
        }

        _report.Info(label, $"{FormatDecimal(value.Value)} ({FormatPercent(value.Value)}).");
    }

    private void ReportRows(JsonElement root, string path, string label)
    {
        int? value = JsonHelpers.GetInt(root, path);
        if (!value.HasValue)
        {
            _report.Warn(label, $"Missing row count: {path}.");
            return;
        }

        _report.Info(label, value.Value.ToString("N0", CultureInfo.InvariantCulture));
    }

    private static string ResolveProjectRoot(string? requestedRoot)
    {
        if (!string.IsNullOrWhiteSpace(requestedRoot))
        {
            return Path.GetFullPath(requestedRoot);
        }

        DirectoryInfo? current = new(Directory.GetCurrentDirectory());
        while (current is not null)
        {
            if (File.Exists(Path.Combine(current.FullName, "config.json")) &&
                Directory.Exists(Path.Combine(current.FullName, "artifacts")))
            {
                return current.FullName;
            }

            current = current.Parent;
        }

        return Directory.GetCurrentDirectory();
    }

    private static string ResolveArtifactsPath(string root, JsonDocument? runtimeConfig)
    {
        string configuredPath = JsonHelpers.GetString(runtimeConfig?.RootElement, "artifacts_path") ?? "artifacts";
        return Path.IsPathRooted(configuredPath)
            ? configuredPath
            : Path.Combine(root, configuredPath);
    }

    private static string BuildDefaultHealthUrl(JsonDocument? runtimeConfig)
    {
        string host = JsonHelpers.GetString(runtimeConfig?.RootElement, "flask_host") ?? "127.0.0.1";
        int port = JsonHelpers.GetInt(runtimeConfig?.RootElement, "flask_port") ?? 5000;

        if (host == "0.0.0.0")
        {
            host = "127.0.0.1";
        }

        return $"http://{host}:{port.ToString(CultureInfo.InvariantCulture)}/health";
    }

    private static string FormatPercent(double value)
    {
        return (value * 100).ToString("0.00", CultureInfo.InvariantCulture) + "%";
    }

    private static string FormatDecimal(double value)
    {
        return value.ToString("0.###", CultureInfo.InvariantCulture);
    }

    private static string FormatBytes(long value)
    {
        string[] units = ["B", "KB", "MB", "GB"];
        double size = value;
        int unitIndex = 0;

        while (size >= 1024 && unitIndex < units.Length - 1)
        {
            size /= 1024;
            unitIndex++;
        }

        return $"{size.ToString("0.##", CultureInfo.InvariantCulture)} {units[unitIndex]}";
    }

    private static string ToDisplayString(JsonElement element)
    {
        return element.ValueKind == JsonValueKind.String ? element.GetString() ?? string.Empty : element.ToString();
    }
}

internal sealed class DiagnosticReport
{
    private int _okCount;
    private int _warningCount;
    private int _failureCount;

    public void Title(string text)
    {
        Console.WriteLine(text);
        Console.WriteLine(new string('=', text.Length));
    }

    public void Section(string text)
    {
        Console.WriteLine();
        Console.WriteLine(text);
        Console.WriteLine(new string('-', text.Length));
    }

    public void Ok(string label, string detail)
    {
        _okCount++;
        Write("OK", label, detail);
    }

    public void Warn(string label, string detail)
    {
        _warningCount++;
        Write("WARN", label, detail);
    }

    public void Fail(string label, string detail)
    {
        _failureCount++;
        Write("FAIL", label, detail);
    }

    public void Info(string label, string detail)
    {
        Write("INFO", label, detail);
    }

    public int Summary()
    {
        Console.WriteLine();
        Console.WriteLine("Summary");
        Console.WriteLine("-------");
        Console.WriteLine($"OK: {_okCount}, warnings: {_warningCount}, failures: {_failureCount}");

        if (_failureCount > 0)
        {
            Console.WriteLine("Result: FAIL");
            return 1;
        }

        Console.WriteLine(_warningCount > 0 ? "Result: PASS with warnings" : "Result: PASS");
        return 0;
    }

    private static void Write(string status, string label, string detail)
    {
        Console.WriteLine($"[{status}] {label}: {detail}");
    }
}

internal sealed class CliOptions
{
    public string? Root { get; private set; }
    public string? HealthUrl { get; private set; }
    public bool SkipHealth { get; private set; }
    public bool ShowHelp { get; private set; }
    public string? Error { get; private set; }

    public static CliOptions Parse(string[] args)
    {
        CliOptions options = new();

        for (int i = 0; i < args.Length; i++)
        {
            string arg = args[i];
            switch (arg)
            {
                case "--help":
                case "-h":
                    options.ShowHelp = true;
                    break;
                case "--root":
                    if (!TryReadValue(args, ref i, out string? root))
                    {
                        options.SetError("--root requires a path value.");
                        return options;
                    }

                    options.Root = root;
                    break;
                case "--health-url":
                    if (!TryReadValue(args, ref i, out string? healthUrl))
                    {
                        options.SetError("--health-url requires a URL value.");
                        return options;
                    }

                    options.HealthUrl = healthUrl;
                    break;
                case "--skip-health":
                    options.SkipHealth = true;
                    break;
                default:
                    options.SetError($"Unknown argument: {arg}");
                    return options;
            }
        }

        return options;
    }

    private void SetError(string error)
    {
        Error = error;
        ShowHelp = true;
    }

    private static bool TryReadValue(string[] args, ref int index, out string? value)
    {
        if (index + 1 >= args.Length || args[index + 1].StartsWith("--", StringComparison.Ordinal))
        {
            value = null;
            return false;
        }

        index++;
        value = args[index];
        return true;
    }
}

internal static class JsonHelpers
{
    public static bool TryGetElement(JsonElement? root, string path, out JsonElement element)
    {
        if (!root.HasValue)
        {
            element = default;
            return false;
        }

        element = root.Value;
        foreach (string part in path.Split('.'))
        {
            if (element.ValueKind != JsonValueKind.Object || !element.TryGetProperty(part, out JsonElement child))
            {
                element = default;
                return false;
            }

            element = child;
        }

        return true;
    }

    public static string? GetString(JsonElement? root, string path)
    {
        if (!TryGetElement(root, path, out JsonElement element))
        {
            return null;
        }

        return element.ValueKind == JsonValueKind.String ? element.GetString() : element.ToString();
    }

    public static double? GetDouble(JsonElement? root, string path)
    {
        if (!TryGetElement(root, path, out JsonElement element))
        {
            return null;
        }

        if (element.ValueKind == JsonValueKind.Number && element.TryGetDouble(out double value))
        {
            return value;
        }

        if (element.ValueKind == JsonValueKind.String &&
            double.TryParse(element.GetString(), NumberStyles.Float, CultureInfo.InvariantCulture, out value))
        {
            return value;
        }

        return null;
    }

    public static int? GetInt(JsonElement? root, string path)
    {
        if (!TryGetElement(root, path, out JsonElement element))
        {
            return null;
        }

        if (element.ValueKind == JsonValueKind.Number && element.TryGetInt32(out int value))
        {
            return value;
        }

        if (element.ValueKind == JsonValueKind.String &&
            int.TryParse(element.GetString(), NumberStyles.Integer, CultureInfo.InvariantCulture, out value))
        {
            return value;
        }

        return null;
    }

    public static bool? GetBool(JsonElement? root, string path)
    {
        if (!TryGetElement(root, path, out JsonElement element))
        {
            return null;
        }

        return element.ValueKind switch
        {
            JsonValueKind.True => true,
            JsonValueKind.False => false,
            JsonValueKind.String when bool.TryParse(element.GetString(), out bool value) => value,
            _ => null
        };
    }
}
