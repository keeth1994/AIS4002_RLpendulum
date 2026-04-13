# AIS4002_RLpendulum

This project currently runs with standard Python only and does not require any third-party packages.

## Run locally

1. Clone the repository:

```bash
git clone <your-repo-url>
cd AIS4002_RLpendulum
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

3. Install requirements:

```bash
pip install -r requirements.txt
```

4. Run the program:

```bash
python main.py
```

## Notes

- `requirements.txt` is intentionally minimal because the project does not currently use external Python packages.
- If you add libraries later, install them in your environment and then update `requirements.txt`.
