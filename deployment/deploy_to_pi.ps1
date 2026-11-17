Param(
    [string]$PiHost = "",
    [string]$PiUser = "pi",
    [int]$Port = 22,
    [string]$RemoteDir = "~/SafeStep_Deploy",
    [switch]$RunInstall,
    [switch]$IncludeDatasets,
    [switch]$DryRun
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

$ArgsList = @("$ScriptDir\deploy_to_pi.py", "--user", $PiUser, "--port", "$Port", "--remote-dir", $RemoteDir)
if ($PiHost -ne "") {
    $ArgsList += @("--host", $PiHost)
} else {
    $ArgsList += "--dry-run"
}
if ($RunInstall) { $ArgsList += "--run-install" }
if ($IncludeDatasets) { $ArgsList += "--include-datasets" }
if ($DryRun) { $ArgsList += "--dry-run" }

Push-Location $ProjectRoot
python @ArgsList
Pop-Location
