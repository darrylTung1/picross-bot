# Picross Touch Bot

A bot that solves [Picross Touch](https://store.steampowered.com/app/476460/Picross_Touch/) puzzles by itself: it finds the puzzle on your screen, reads the clue numbers, solves the puzzle, and fills in the answer with the mouse. In auto mode it plays level after level.

## Features

* **Finds the board automatically.** Any puzzle size, anywhere on screen, main game or Steam Workshop levels.
* **Learns the game's digits as you play.** The first few boards ask you what some digit shapes are; after that it reads them on its own, including at board sizes it hasn't seen.
* **Never fills a wrong answer.** Every reading is checked against the puzzle's own logic (the clues must add up and the puzzle must solve) before anything is clicked.
* **Solves in milliseconds,** even 40×40, then drags along each run of filled cells.
* **Auto mode** solves each new puzzle and clicks through to the next level.
* **Safe on half-done boards.** It reads which cells are already filled and only fills the rest.

## Setup

Needs Python 3.9 or newer. Developed and tested on Windows.

```
git clone https://github.com/<your-username>/picross-bot.git
cd picross-bot
pip install -r requirements.txt
python selftest.py
```

The self-test plays mock boards drawn in the game's style, so it needs neither the game nor your screen. It should end with `All good.`

## Usage

Start the bot and leave its terminal open:

```
python picross\_bot.py
```

Then switch to the game and use these hotkeys:

|Key|What it does|
|-|-|
|**F8**|Find, read, solve and fill the puzzle on screen|
|**F7**|Same, but first show the clues it read so you can fix any|
|**F10**|Auto mode on/off: solve each new puzzle, and click taught buttons to move on|
|**F6**|Teach a button: hover over it (e.g. *Next*) and press F6|
|**F9**|Stop (or fling the mouse into a screen corner)|

Press Ctrl+C in the terminal to quit.

The whole board, including all clues, must be visible on screen.

### Teaching it the digits

At first the bot doesn't know the game's font. When it meets a digit shape it can't place, it beeps, draws the shape in the terminal and asks what it is:

* Type just the digit drawn. For the 1 of a 13, type `1`.
* If two digits are drawn as one shape, type both, e.g. `11`.
* Type `x` for anything that isn't a number, or press Enter to skip.

What it learns is saved in `picross\_digits.npz`. Expect a handful of questions on your first couple of boards and very few after that.

### Fixing a misread

If the clues it read don't add up, it lists every row and column as read and lets you correct them. Type `c4 2 1` to say column 4 is "2 1", or `r10 0` for an empty row 10, then press Enter. It learns from your fix. **F7** opens the same editor on any board.

### Auto mode

1. Finish a puzzle, then hover over the button that moves on (e.g. *Next*) and press **F6**. Do the same for every button in the chain between levels.
2. Press **F10**. The bot checks the screen every second, solves any unsolved puzzle, and otherwise clicks whichever taught button it sees.

Only teach buttons that appear *between* levels, not ones that are always on screen like menu buttons.

### Options

|Option|Default|Meaning|
|-|-|-|
|`--delay`|`0.02`|Seconds between mouse moves. The game checks the mouse about every 17 ms, so much lower can skip cells.|
|`--interval`|`1.0`|Auto mode: seconds between looks at the screen|
|`--click`||Click each cell instead of dragging|
|`--dry-run`||Solve and print the answer without touching the mouse|
|`--size`||Skip grid-size detection, e.g. `--size 15x15`|

After every F8 it saves `picross\_last.png`, showing what it read and the solution. Open it if something looks wrong.

## How it works

1. **Find the board.** Look for the largest rectangular black outline on screen (the board's frame), then for the first thick lines running the full height and width inside it, which separate the clues from the grid.
2. **Count the cells.** Average the brightness down each column of pixels. Grid lines show up as dips, and the typical spacing between dips gives the cell size (a *projection profile*).
3. **Cut out the digits.** For each clue cell, mark pixels that differ from the cell's background colour, erase the grid lines, and group touching pixels into blobs. Digits that touch each other are split along the boundary between their white fills.
4. **Describe each digit by its outline.** This font draws white digits with thick black outlines that also fill the holes, so only the dark outline tells a 3 from an 8.
5. **Recognise it.**

   * **Exact match:** compare with stored examples of the same size using *chamfer distance* (how far each outline pixel is from the nearest outline pixel of the other).
   * **Classifier:** otherwise, a classifier (HOG features + logistic regression, trained on everything learned so far) gives its best guess. The guess counts only if the digit also looks like that digit's stored examples.
   * **Puzzle check:** every guess is provisional until the clues add up and the puzzle solves. Then it's learned at the new size.
   * **Deduction:** when up to three shapes are unknown, it tries every digit for them and keeps the only combination that works. It asks you only when that fails.
6. **Solve.** For each row or column, dynamic programming finds every cell forced by its clues and what's already known. This repeats across rows and columns until nothing changes, with backtracking if it gets stuck.
7. **Fill.** Drag along each run of filled cells, choosing rows or columns, whichever needs fewer drags.

## Files it creates

These stay on your machine and are listed in `.gitignore`:

|File|What it is|
|-|-|
|`picross\_digits.npz`|The digits it has learned|
|`picross\_buttons/`|Pictures of the buttons you taught|
|`picross\_capture.png`|The last screenshot it took|
|`picross\_last.png`|What it read from that screenshot|

## Limitations

* Built for Picross Touch's look. Other picross games would need changes to the board finding and digit reading.
* Tested on Windows. The libraries also support macOS and Linux, but screen scaling and permissions differ there.
* Very small digits on huge boards may need a few more questions.
* If a cell is filled or crossed out where it shouldn't be, the bot stops rather than clicking it. Reset the puzzle and try again.

## Notes

Not affiliated with the developers of Picross Touch. It's a nonsense project, so please don't use it for leaderboards or speedruns.

## License

[MIT](LICENSE)

