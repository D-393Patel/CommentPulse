param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("api", "stack", "worker", "test", "test-unit", "test-redis", "labeling-queue", "labeling-bootstrap", "labeling-review", "labeling-audit", "load-test", "promote-baseline")]
    [string]$Task
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"

switch ($Task) {
    "api" {
        & $python "$root\app.py"
    }
    "stack" {
        Push-Location $root
        try {
            docker compose up --build
        }
        finally {
            Pop-Location
        }
    }
    "worker" {
        & $python "$root\worker.py"
    }
    "test" {
        & $python -m unittest discover -s "$root\tests" -v
    }
    "test-unit" {
        & $python -m unittest discover -s "$root\tests" -p "test_app.py" -v
    }
    "test-redis" {
        $env:RUN_REDIS_INTEGRATION = "1"
        if (-not $env:REDIS_URL) {
            $env:REDIS_URL = "redis://localhost:6379/0"
        }
        & $python -m unittest discover -s "$root\tests" -p "test_redis_integration.py" -v
    }
    "labeling-queue" {
        & $python "$root\src\data\dataset_curation.py" prepare `
            --input "$root\data.json" `
            --output "$root\data\external\youtube_comments_labeling_queue.csv" `
            --dataset-name "youtube_capture" `
            --existing-labeled "$root\data\external\youtube_comments_labeled.csv" `
            --batch-size 100 `
            --add-model-suggestions
    }
    "labeling-bootstrap" {
        & $python "$root\src\data\dataset_curation.py" bootstrap `
            --input "$root\data\external\youtube_comments_labeling_queue.csv" `
            --output "$root\data\external\youtube_comments_pseudo_labeled.csv" `
            --min-confidence 0.75
    }
    "labeling-review" {
        & $python "$root\src\data\dataset_curation.py" review-prepare `
            --input "$root\data\external\youtube_comments_pseudo_labeled.csv" `
            --output "$root\data\external\youtube_comments_review_queue.csv"
    }
    "labeling-audit" {
        & $python "$root\src\data\dataset_curation.py" audit `
            --input "$root\data\external\youtube_comments_labeled.csv"
    }
    "load-test" {
        $stdout = Join-Path $root "reports\performance\load_test_server_stdout.log"
        $stderr = Join-Path $root "reports\performance\load_test_server_stderr.log"
        New-Item -ItemType Directory -Force -Path (Split-Path $stdout) | Out-Null
        $psi = New-Object System.Diagnostics.ProcessStartInfo
        $psi.FileName = $python
        $psi.Arguments = "`"$root\app.py`""
        $psi.WorkingDirectory = $root
        $psi.UseShellExecute = $false
        $psi.RedirectStandardOutput = $true
        $psi.RedirectStandardError = $true
        $server = New-Object System.Diagnostics.Process
        $server.StartInfo = $psi
        $null = $server.Start()
        try {
            $healthy = $false
            for ($i = 0; $i -lt 60; $i++) {
                try {
                    $response = Invoke-WebRequest -Uri "http://127.0.0.1:5000/health" -UseBasicParsing -TimeoutSec 2
                    if ($response.StatusCode -eq 200) {
                        $healthy = $true
                        break
                    }
                }
                catch {
                    Start-Sleep -Milliseconds 500
                }
            }
            if (-not $healthy) {
                throw "Local API did not become healthy in time."
            }
            & $python "$root\scripts\load_test.py" --base-url "http://127.0.0.1:5000"
        }
        finally {
            if ($server -and -not $server.HasExited) {
                Stop-Process -Id $server.Id -Force
                $server.WaitForExit()
            }
            if ($server) {
                $server.StandardOutput.ReadToEnd() | Set-Content -Path $stdout
                $server.StandardError.ReadToEnd() | Set-Content -Path $stderr
            }
        }
    }
    "promote-baseline" {
        & $python "$root\src\model\promote_baseline.py"
    }
}
