# reference-bench.ps1 — the same models, the same card, the real NVIDIA driver: reference numbers for the shim.
#   .\reference-bench.ps1 prepare        download + unpack the latest prebuilt llama.cpp and stable-diffusion.cpp CUDA builds
#   .\reference-bench.ps1 bench          llama-bench every GGUF under $Models (tg128 / pp256, like tools/nv_shim_step.sh bench),
#                                        then SDXL-Turbo / SD 1.5 / Z-Image-Turbo with the same seeds and steps as the Mac runs
#   .\reference-bench.ps1 all
# Results land in $Out\reference-<timestamp>.md, one table per tool, plus nvidia-smi's driver/clock lines for provenance.
param([string]$Mode = "all")
$ErrorActionPreference = "Stop"
$Root   = "$env:USERPROFILE\egpu-reference"
$Models = "$Root\models"        # put the GGUFs / safetensors here (copy from the Mac: models\ and models\sd\)
$Out    = "$Root\results"
New-Item -ItemType Directory -Force -Path $Root, $Models, $Out | Out-Null

function Get-Release($repo, $pattern) {
  # "latest" is not always the release that carries binaries (llama.cpp's versioned tags carry only a nightly pointer),
  # so scan the most recent releases for the first one with a matching asset
  $rels = Invoke-RestMethod "https://api.github.com/repos/$repo/releases?per_page=30" -Headers @{ "User-Agent" = "egpu-reference" }
  $rel = $null; $asset = $null
  foreach ($r in $rels) { $a = $r.assets | Where-Object { $_.name -match $pattern } | Select-Object -First 1; if ($a) { $rel = $r; $asset = $a; break } }
  if (-not $asset) { throw "no asset matching $pattern in the last 30 releases of $repo (latest: $($rels[0].tag_name): $($rels[0].assets.name -join ', '))" }
  $zip = "$Root\$($asset.name)"
  if (-not (Test-Path $zip)) { Write-Host "downloading $($asset.name) ($([int]($asset.size/1MB)) MB)"; Invoke-WebRequest $asset.browser_download_url -OutFile $zip }
  return @{ zip = $zip; tag = $rel.tag_name; name = $asset.name }
}

if ($Mode -eq "prepare" -or $Mode -eq "all") {
  # llama.cpp ships the CUDA binaries and the CUDA runtime DLLs as two zips; both go into one folder
  $l = Get-Release "ggml-org/llama.cpp" "^llama-.*-bin-win-cuda.*-x64\.zip$"
  $c = Get-Release "ggml-org/llama.cpp" "^cudart-llama-bin-win-cuda.*-x64\.zip$"
  Expand-Archive -Force $l.zip "$Root\llama.cpp"; Expand-Archive -Force $c.zip "$Root\llama.cpp"
  # stable-diffusion.cpp also ships binaries and the CUDA runtime as two zips
  $s  = Get-Release "leejet/stable-diffusion.cpp" "^sd-.*win.*cu(da)?12.*x64.*\.zip$"
  $sc = Get-Release "leejet/stable-diffusion.cpp" "^cudart-sd-.*win.*cu(da)?12.*x64.*\.zip$"
  Expand-Archive -Force $s.zip "$Root\sd.cpp"; Expand-Archive -Force $sc.zip "$Root\sd.cpp"
  # models that already live in LM Studio's folders on this machine: copy, do not re-download
  $lm = "$env:USERPROFILE\.lmstudio\models"
  foreach ($f in @("unsloth\gemma-4-26B-A4B-it-qat-GGUF\gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf", "lmstudio-community\Qwen3.5-35B-A3B-GGUF\Qwen3.5-35B-A3B-Q4_K_M.gguf",
                   "bartowski\nvidia_NVIDIA-Nemotron-Nano-9B-v2-GGUF\nvidia_NVIDIA-Nemotron-Nano-9B-v2-Q6_K.gguf", "lmstudio-community\gemma-4-12B-it-GGUF\gemma-4-12B-it-Q4_K_M.gguf")) {
    $src = Join-Path $lm $f; $dst = Join-Path $Models (Split-Path $f -Leaf)
    if ((Test-Path $src) -and -not (Test-Path $dst)) { Write-Host "copying $(Split-Path $f -Leaf) from LM Studio"; Copy-Item $src $dst }
  }
  "prepared: llama.cpp $($l.tag) ($($l.name)), stable-diffusion.cpp $($s.tag) ($($s.name))" | Tee-Object "$Root\prepared.txt"
  Get-ChildItem -Recurse $Root\llama.cpp -Filter llama-bench.exe | Select-Object -First 1 -ExpandProperty FullName
  Get-ChildItem -Recurse $Root\sd.cpp -Include sd-cli.exe, sd.exe | Select-Object -First 1 -ExpandProperty FullName
}

if ($Mode -eq "bench" -or $Mode -eq "all") {
  $ts = Get-Date -Format "yyyyMMdd-HHmmss"; $report = "$Out\reference-$ts.md"
  $bench = Get-ChildItem -Recurse $Root\llama.cpp -Filter llama-bench.exe | Select-Object -First 1 -ExpandProperty FullName
  $sd    = Get-ChildItem -Recurse $Root\sd.cpp -Include sd-cli.exe, sd.exe | Select-Object -First 1 -ExpandProperty FullName
  "# Reference run $ts (Windows, real NVIDIA driver)`n" | Out-File $report
  "## provenance`n``````" | Out-File $report -Append
  nvidia-smi --query-gpu=name,driver_version,pcie.link.gen.current,pcie.link.width.current,clocks.max.sm,memory.total --format=csv | Out-File $report -Append
  Get-Content "$Root\prepared.txt" | Out-File $report -Append; "``````" | Out-File $report -Append
  "## llama-bench (-ngl 99 -p 256 -n 128 -r 3, same as the Mac runs)`n" | Out-File $report -Append
  Get-ChildItem $Models -Filter *.gguf | Where-Object { $_.Name -notmatch '^mmproj|^mtp-|Qwen3-4B-Instruct' } | ForEach-Object {
    Write-Host "llama-bench $($_.Name)"
    & $bench -m $_.FullName -ngl 99 -p 256 -n 128 -r 3 -o md 2>$null | Out-File $report -Append
  }
  # MTP speculative decode, the number the shim's 100 tok/s is compared against
  $spec = Get-ChildItem -Recurse $Root\llama.cpp -Filter llama-speculative-simple.exe | Select-Object -First 1 -ExpandProperty FullName
  $tgt = "$Models\Qwen3.8-27B-UD-Q4_K_M.gguf"; $mtp = "$Models\mtp-Qwen3.8-27B-Q4_0.gguf"
  if ($spec -and (Test-Path $tgt) -and (Test-Path $mtp)) {
    "## MTP (llama-speculative-simple, draft-mtp, depth 4, greedy, n=256)`n``````" | Out-File $report -Append
    & $spec -m $tgt -md $mtp -ngl 99 -ngld 99 --spec-type draft-mtp --spec-draft-n-max 4 -n 256 --temp 0 -p "The three most important things to know about the Thunderbolt bus are" 2>&1 |
      Select-String -Pattern "decoded|n_accept|accept " | Out-File $report -Append
    "``````" | Out-File $report -Append
  }
  if ($sd) {
    "## image generation (same seeds/steps as the Mac runs)`n``````" | Out-File $report -Append
    $p = "a photograph of a red fox sitting on a mossy log in a sunlit forest, detailed fur"
    $runs = @(
      @{ n = "SDXL-Turbo 4 steps 512"; a = @("-m", "$Models\sd\sd_xl_turbo_1.0_fp16.safetensors", "--steps", "4", "--cfg-scale", "1.0", "--sampling-method", "euler", "-W", "512", "-H", "512") },
      @{ n = "SD 1.5 20 steps 512";    a = @("-m", "$Models\sd\v1-5-pruned-emaonly.safetensors", "--steps", "20", "-W", "512", "-H", "512") },
      @{ n = "Z-Image-Turbo 8 steps 1024"; a = @("--diffusion-model", "$Models\sd\z_image_turbo-Q8_0.gguf", "--vae", "$Models\sd\flux-ae.safetensors", "--llm", "$Models\sd\Qwen3-4B-Instruct-2507-Q8_0.gguf", "--steps", "8", "--cfg-scale", "1.0", "--sampling-method", "euler", "-W", "1024", "-H", "1024") }
    )
    foreach ($r in $runs) {
      Write-Host "sd: $($r.n)"
      $lines = & $sd @($r.a) -p $p -s 42 -o "$Out\$($r.n -replace ' ','_')-$ts.png" 2>&1 | Select-String -Pattern "sampling completed|decode_first_stage|generate_image completed"
      "$($r.n): $($lines -join ' | ')" | Out-File $report -Append
    }
    "``````" | Out-File $report -Append
  }
  Write-Host "report: $report"; Get-Content $report
}
