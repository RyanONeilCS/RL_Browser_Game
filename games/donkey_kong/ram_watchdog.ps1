# RAM watchdog for a long training run (Donkey Kong v1, 2026-10-07).
# Every 5 minutes: logs free RAM and commit to runs/v1_ram.log. Stops the training (the whole process tree:
# the game processes and browsers too) if memory gets dangerous, and writes an ALERT line. A WARN line if available
# RAM fell by more than 2 GB in the last hour (not stopped).
#   powershell -ExecutionPolicy Bypass -File games\donkey_kong\ram_watchdog.ps1
# Killing the run loses at most the steps since the last checkpoint (every ~50k steps); the final save doesn't run.
param(
    [int]$IntervalSec = 300,
    [int]$MinAvailableMB = 1500,     # stop below this much available RAM
    [int]$MinCommitFreeMB = 3000,    # stop below this much commit left (when commit is full, programs crash)
    [string]$Log = "$PSScriptRoot\runs\v1_ram.log"
)

function Find-Training {
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -like '*donkey_kong*train.py*' } |
        Sort-Object CreationDate | Select-Object -First 1
}

function Write-Log($text) {
    # Retried: while another program has the log open (e.g. tail -f), a write can fail. Never stop over the log.
    $line = "{0}  {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm"), $text
    for ($i = 0; $i -lt 10; $i++) {
        try { [System.IO.File]::AppendAllText($Log, $line + "`r`n"); return } catch { Start-Sleep -Milliseconds 500 }
    }
}

$history = @()   # {T = time, MB = available} for the last hour
Write-Log "watchdog started (stop below $MinAvailableMB MB available or $MinCommitFreeMB MB commit left)"
while ($true) {
    $train = Find-Training
    if (-not $train) { Write-Log "STOPPED: training is not running any more (finished, stopped or crashed). Watchdog ends."; break }

    $avail = [int](Get-Counter '\Memory\Available MBytes').CounterSamples[0].CookedValue
    $os = Get-CimInstance Win32_OperatingSystem
    $commitFree = [int]($os.FreeVirtualMemory / 1KB)
    $commitUsed = [math]::Round(($os.TotalVirtualMemorySize - $os.FreeVirtualMemory) / 1MB, 1)
    $pyMB = [int]((Get-Process python -ErrorAction SilentlyContinue | Measure-Object WorkingSet64 -Sum).Sum / 1MB)
    $brMB = [int]((Get-Process chrome-headless-shell -ErrorAction SilentlyContinue | Measure-Object WorkingSet64 -Sum).Sum / 1MB)
    Write-Log ("ok  available {0} MB, commit used {1} GB ({2} MB left), python {3} MB, game browsers {4} MB" -f $avail, $commitUsed, $commitFree, $pyMB, $brMB)

    $now = Get-Date
    $history = @($history | Where-Object { ($now - $_.T).TotalMinutes -le 60 }) + @([pscustomobject]@{ T = $now; MB = $avail })
    $oldest = $history[0]
    if (($now - $oldest.T).TotalMinutes -ge 50 -and ($oldest.MB - $avail) -gt 2000) {
        Write-Log ("WARN: available RAM fell {0} MB in the last hour (from {1} to {2} MB)" -f ($oldest.MB - $avail), $oldest.MB, $avail)
        $history = @([pscustomobject]@{ T = $now; MB = $avail })   # don't repeat the warning every 5 minutes
    }

    if ($avail -lt $MinAvailableMB -or $commitFree -lt $MinCommitFreeMB) {
        Write-Log ("ALERT: memory too low (available {0} MB, commit left {1} MB). Stopping the training (PID {2})." -f $avail, $commitFree, $train.ProcessId)
        taskkill /PID $train.ProcessId /T /F | Out-Null
        Start-Sleep 5
        if (Find-Training) { Write-Log "ALERT: training still running after taskkill!" } else { Write-Log "ALERT: training stopped. Restart from the newest checkpoint: RESUME = ""latest"" in train.py." }
        break
    }
    Start-Sleep $IntervalSec
}
