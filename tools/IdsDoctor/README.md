# IDS Doctor

`IdsDoctor` is an optional C# diagnostics client for the IDS project.

It is intentionally read-only. It does not import Python modules, modify model artifacts,
write logs, start the Flask server, or change runtime behavior.

## Run

From the repository root:

```powershell
dotnet run --project tools/IdsDoctor
```

With an explicit root:

```powershell
dotnet run --project tools/IdsDoctor -- --root C:\project\ids
```

Skip the optional HTTP health check:

```powershell
dotnet run --project tools/IdsDoctor -- --skip-health
```

## What It Checks

- `config.json` runtime settings.
- Required files under `artifacts/`.
- Raw NSL-KDD data files under `data/raw/`.
- Binary and multiclass evaluation report metrics.
- Multiclass attack-family label mapping.
- Optional Flask `/health` endpoint.

The tool exits with code `1` only when required files or required config values are
missing or malformed. Model-quality concerns and an offline Flask server are reported
as warnings.
