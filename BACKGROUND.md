# Run Scripts in the background and check on them

## Tensorboard

-   Start via full command:

    ```
    tensorboard --logdir /media/jonas/69B577D0C4C25263/MLData/tensorboard --bind_all
    ```

-   Start via shortcut-script (is just a script that contains above line):

    ```
    ~/tensorboard.sh
    ```

-   Visit the URL:
    -   local: http://localhost:6006
    -   remote/external Pc: http://192.168.2.183:6006 (or the corresponding local IP)

## ML-Process in background

-   start the process:
    ```
    nohup /bin/python3 /home/jonas/Github/bachelor-thesis-experiments/managing_imagenet_data/training_script.py &
    ```
    -   `nohup` (No-Holdup) lets it run after terminal closes.
    -   `&` makes it a background process and prints the Process-ID for later
-   monitor running Process (and getting Process-ID)
    ```
    ps -ef | grep python
    ```
    Look for the corresponding Python process. ! If you have workers enabled, there will be multiple processes. They all have the same parent-process except for one. This is the one to kill.
-   kill the process:
    ```
    kill <<pid>>
    ```
