# Data

| File | Games | Columns |
|---|---|---|
| `chess_error_detection_train.csv.gz` | 400,000 | `sequence`, `error_position`, `correct_move` |
| `chess_error_detection_test.csv.gz` | 50,000 | `sequence`, `error_position`, `correct_move` |

- `sequence`: first 40 plies of a real game in SAN, one of the first 10 replaced
- `error_position`: position of the replaced ply, 1 to 10 (the target)
- `correct_move`: the original move; used only to rebuild clean games for pre-training, never as model input

Gzip-compressed CSV, read directly by pandas. Provided with the Deep Learning course, University of Bologna.
