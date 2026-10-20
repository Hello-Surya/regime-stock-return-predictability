# Build the complete LaTeX manuscript with a clean author-year bibliography.
# Run from any location: .\scripts\build_paper.ps1
# Never removes research data or source files.
[CmdletBinding()]
param(
    [switch]$KeepAux
)

$ErrorActionPreference = "Stop"
$paperDir = Join-Path (Split-Path -Parent $PSScriptRoot) "paper"

foreach ($name in @("pdflatex", "bibtex")) {
    if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
        throw "Required TeX executable '$name' not found on PATH."
    }
}

Push-Location $paperDir
try {
    if (-not $KeepAux) {
        # Earlier manuscript builds used the numeric 'plain' BibTeX style.
        # Removing its .aux/.bbl data prevents natbib from forcing numerical
        # citations when the current Chicago author-year .bst is loaded.
        foreach ($name in @("main.aux", "main.bbl", "main.blg", "main.out")) {
            if (Test-Path -LiteralPath $name) {
                Remove-Item -LiteralPath $name -Force
            }
        }
        Write-Host "Cleaned old LaTeX/BibTeX auxiliary files."
    }

    $steps = @(
        @{ Program = "pdflatex"; Arguments = @("-interaction=nonstopmode", "-halt-on-error", "main.tex") },
        @{ Program = "bibtex"; Arguments = @("main") },
        @{ Program = "pdflatex"; Arguments = @("-interaction=nonstopmode", "-halt-on-error", "main.tex") },
        @{ Program = "pdflatex"; Arguments = @("-interaction=nonstopmode", "-halt-on-error", "main.tex") }
    )

    for ($i = 0; $i -lt $steps.Count; $i++) {
        $step = $steps[$i]
        Write-Host ("[{0}/4] Running {1}..." -f ($i + 1), $step.Program)
        & $step.Program @($step.Arguments)
        if ($LASTEXITCODE -ne 0) {
            throw ("{0} failed with exit code {1}. See paper/main.log and paper/main.blg." -f $step.Program, $LASTEXITCODE)
        }
    }

    if (-not (Test-Path -LiteralPath "main.pdf")) {
        throw "The TeX commands completed but paper/main.pdf was not produced."
    }
    Write-Host "PASS: paper/main.pdf compiled with a fresh author-year bibliography."
}
finally {
    Pop-Location
}
