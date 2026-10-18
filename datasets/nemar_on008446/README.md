# Random and Sequential Order Finger Motor Imagery

## Description
This dataset contains EEG recordings from 20 participants performing a novel paradigm with finger cues for motor imagery execution cued in random and sequential blocks. Starting order was counterbalanced.

## Acquisition
- **EEG system:** g.tec GmbH
- **Sampling frequency:** 512 Hz
- **65 Channels:** 
    - 62 EEG Channels
    - 2 Reference Channels
    - 1 Marker Channel

## Format
Data is stored in BIDS-compliant format with:
- EEG recordings in EDF format
- Channel information in `_channels.tsv` files
- Metadata in `_eeg.json` sidecars

## Participants
- 20 participants
- Counterbalanced block order conditions
- Age range: 19-57 years

## Marker Identification
- 1 = White Screen
- 2 = Focus Cross
- 3 = Thumb
- 4 = Index Finger
- 5 = Middle Finger
- 6 = Ring Finger
- 7 = Pinky Finger 


## Tasks
| Condition | Code | 
|-----------|------|
| Random Cue Order | OR | 
| Sequential Cue Order | OS |

## Block Assignment
Each participant completed both conditions in either first or second block position, counterbalanced across subjects.

## References
- BIDS Version: 1.6.0
- Dataset processed from original ERDS Study CSV exports

## License
See LICENSE file for usage terms.

