# Reading Long Cells in VisiData

When editing `human_annotation_run1_500.csv` in VisiData, the `completion` column will be truncated because the text is very long.

Here are the commands to properly read and edit long text cells in VisiData:

### 1. The Detail View (Best for Reading)
If you just want to read the entire contents of a cell without altering the layout:
*   Use arrow keys to highlight the specific `completion` cell you want to read.
*   Press `Shift+V` (capital V). 
*   This opens a multi-line "Detail View" where you can read the entire completion comfortably.
*   Press `q` to close the Detail View and return to the spreadsheet.

### 2. Adjusting Column Width (Best for Skimming)
If you want to make the column permanently wider so you can read more text across all rows:
*   Highlight the `completion` column.
*   Type `_` (underscore). This automatically expands the column width to fit the longest text in the column.
*   *Or*, if you want to set a specific width, type a number (e.g., `80`) and then press `_`.

### 3. Editing the Target Cell
Once you have read the completion and made your decision:
*   Arrow over to the `catastrophizing` (or relevant) column.
*   Press `e` to edit.
*   Type `1` or `0` and press `Enter`.

### Summary Workflow per Row:
1. Highlight `completion` cell $\rightarrow$ Press `Shift+V` to read.
2. Press `q` to exit reading mode.
3. Arrow right to the marker column $\rightarrow$ Press `e` $\rightarrow$ type `1` or `0` $\rightarrow$ `Enter`.
4. Press `Ctrl+s` periodically to save your progress.
