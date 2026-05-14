# Reproducible recording of the lite-runner README demo.
#
# Requires `vhs` (https://github.com/charmbracelet/vhs):
#   brew install vhs
#
# Rebuild the GIF with:
#   just record-demo
# or directly:
#   vhs examples/demo.tape
#
# Output: examples/demo.gif

Output examples/demo.gif

Set Shell bash
Set FontSize 16
Set Width 1100
Set Height 650
Set TypingSpeed 40ms
Set Padding 12

Type "./examples/run_example.py --prompt 'a cat walking on the beach' --no-wandb"
Sleep 500ms
Enter
Sleep 6s

Type "ls ~/lite_runs/lite-runner/$(ls -t ~/lite_runs/lite-runner | head -1)/"
Sleep 300ms
Enter
Sleep 2s

Type "cat ~/lite_runs/lite-runner/$(ls -t ~/lite_runs/lite-runner | head -1)/run_info.json | head -40"
Sleep 300ms
Enter
Sleep 4s
