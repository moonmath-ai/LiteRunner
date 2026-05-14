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

Output docs/demo.gif

Set Shell bash
Set FontSize 16
Set Width 1100
Set Height 650
Set TypingSpeed 40ms
Set Padding 12

Type "./examples/run_example.py --prompt 'a cat walking on the beach'"
Sleep 500ms
Enter
Sleep 1s

# Step through the 4 TUI prompts:
# image[path], image[start_frame] — accept defaults.
# image[strength] — edit the default 0.8 → 0.9 to show interactive editing.
# seed — accept default.
Enter
Sleep 400ms
Enter
Sleep 400ms
Backspace
Sleep 200ms
Type "9"
Sleep 400ms
Enter
Sleep 400ms
Enter
Sleep 10s
