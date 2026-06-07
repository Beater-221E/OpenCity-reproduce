# T4: OpenCity-Plus pretrain (Windows PowerShell)
Set-Location "$PSScriptRoot\..\model"

python Run.py `
  -mode pretrain `
  -model OpenCity `
  -save_pretrain_path OpenCity-plus2.0.pth `
  -batch_size 4 `
  --embed_dim 512 `
  --skip_dim 512 `
  --enc_depth 6
