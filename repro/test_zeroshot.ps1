# T3/T5: Zero-shot test (Windows PowerShell)
# Set exactly ONE dataset in ..\conf\general_conf\pretrain.conf before running
param(
    [string]$Checkpoint = "OpenCity-plus.pth"
)

Set-Location "$PSScriptRoot\..\model"

python Run.py `
  -mode test `
  -model OpenCity `
  -load_pretrain_path $Checkpoint `
  -batch_size 2 `
  --embed_dim 512 `
  --skip_dim 512 `
  --enc_depth 6
