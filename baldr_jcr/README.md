# baldr RTC
Jesse's implementation of the baldr RTC.

## Todo:
 - [ ] graceful exit of program (e.g., via commander method)

## RTC Logic
![rtc diagram](./baldr_control_logic.svg)

## Compliance Matrix
### Data Objects
| Data Object | Servo Loop | Commander | Supervisor |
|--|:--:|:--:|:--:|
|`meas_offset`|:x::white_check_mark:|:x:|:x:|
|`delay`|:x:|:x:|:x:|
|`meas_to_mode`|:x:|:x:|:x:|
|`filter_coeff_in`|:x:|:x:|:x:|
|`filter_coeff_out`|:x:|:x:|:x:|
|`mode_offset`|:x:|:x:|:x:|
|`windup_max`|:x:|:x:|:x:|
|`windup_min`|:x:|:x:|:x:|
|`mode_to_com`|:x:|:x:|:x:|
|`com_max`|:x:|:x:|:x:|
|`com_min`|:x:|:x:|:x:|
|`com_dist`|:x:|:x:|:x:|
|`com_offset`|:x:|:x:|:x:|
|`com_to_meas`|:x:|:x:|:x:|

### HRTC Pipeline
| Task | Implemented | Tested |
|--|:--:|:--:|
|`read_shm`|:white_check_mark:|:x:|
|`calibrate_frame`|:white_check_mark:|:x:|
|`compute_pol_meas`|:white_check_mark:|:x:|
|`reconstruct_modes`|:white_check_mark:|:x:|
|`filter_modes`|:white_check_mark:|:x:|
|`project_com`|:white_check_mark:|:x:|
|`clip_com`|:white_check_mark:|:x:|
|`inject_disturb`|:white_check_mark:|:x:|
|`write_shm`|:white_check_mark:|:x:|
|`remove_offset`|:white_check_mark:|:x:|
|`inject_dm_signal`|:white_check_mark:|:x:|