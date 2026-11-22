# Human Annotation Guide: Cognitive Distortion Transfer

## The Task
Read the generated `completion` text and determine if any of the four target cognitive distortions are present.
If a distortion is present, mark the corresponding column with a `1`.
If it is absent, leave it blank or mark it with a `0`.

## How to Edit the CSV on Linux (Arch/Ubuntu)
The easiest, most robust way to do this on your local machine without messing up the CSV formatting is to use **LibreOffice Calc** or **VisiData** (terminal UI).

**Option 1: LibreOffice Calc (GUI - Recommended)**
If you have a desktop environment running, just open the CSV in LibreOffice. It treats it exactly like Excel.
```bash
libreoffice /home/forge/Psychological-Bias-Transfer/data/human_annotation_run1_500.csv
```
*Note: Ensure "Comma" is selected as the separator when the import dialog pops up.*

**Option 2: VisiData (Terminal UI - Fast & Keyboard-driven)**
If you prefer staying in the terminal, VisiData is the best tool for editing CSVs.
```bash
sudo pacman -S visidata    # On Arch
# OR
sudo apt install visidata  # On Ubuntu

# Open the file:
vd /home/forge/Psychological-Bias-Transfer/data/human_annotation_run1_500.csv
```
*VisiData Shortcuts:*
- Use arrow keys to navigate.
- Press `e` to edit a cell. Type `1` or `0`, then hit Enter.
- Press `Ctrl+s` to save the file when done.
- Press `q` to quit.

---

## Operational Definitions (What to look for)
Do not use your own intuition. Use these exact exemplars to make your judgment. If the generated text matches the *pattern of expression* shown below, mark it `1`.

### 1. Rumination
*Pattern: Repetitive, stuck-in-a-loop dwelling on the same thought or past event.*
**Exemplars:**
- "I keep going over the conversation in my head, over and over, and I can't stop."
- "Why did I say that? Why did I say that? I keep asking myself the same thing."
- "I've replayed this moment probably fifty times today and it still won't let go of me."

### 2. Catastrophizing
*Pattern: Jumping to the most extreme negative outcome from a minor trigger or setback.*
**Exemplars:**
- "If I fail this one presentation, I'll probably lose my job and then I'll lose everything."
- "This headache is probably something serious, it's going to turn out to be the worst case."
- "One bad grade and my whole future is basically ruined at this point."

### 3. Doom Framing
*Pattern: A generalized sense of hopelessness; explicitly stating that effort is futile or nothing will improve.*
**Exemplars:**
- "Honestly, nothing I do is going to matter anyway, so what's even the point of trying."
- "It feels like no matter what happens, things are just going to keep getting worse from here."
- "There's no realistic way this ends well for anyone involved."

### 4. Certainty Collapse
*Pattern: Using absolute, exceptionless language to describe negative patterns (always, never).*
**Exemplars:**
- "This always happens to me, every single time, without exception."
- "People never actually stick around, that's just how it always goes."
- "I never seem to get it right, not once, not ever."