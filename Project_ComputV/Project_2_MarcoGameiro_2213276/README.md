

Instructions for test of second project - Computational Vision:


The developed project allows to execute from guis_project.py (which refers to the menu itself), but also from each one of the three modules.

- Running it from menu, the following window should open:

![menu](Report_Images/menu.png)

In this case, one of the python modules can be called, depending on the selected option:

- first_task_by_menu.py
- sec_task_by_menu.py
- extra_task_by_menu.py


- When one of the four buttons is selected, a new python file (linked to the option) is executed, and the output results are drained to a new GUI.

- Only the two main functionalities require user input.
    - *For the first one, just one parameter is required:*
        - ## Bag filename ## - [CV_D435_20201104_162148.bag]

        [.../first_task.py -i .../bag_filename.bag]

    - *For the second one, three parameters are required:*
        - ## Bag filename ## - [CV_D435_20201104_162148.bag]

        - ## Initial number of people inside the lab ## - [20]

        - ## CSV filename ## - [csv_file]

        [.../second_task.py -i .../bag_filename.bag -pi init_people_inside -csv csv_filename]

- The extra task, related to camera calibration, can be tested by running extra_task.py script. This will generate a CSV metadata file, similar to the one given by the user, but with the output calibration parameters. 

- If we want to execute each one of the modules separately, we can do it by the command line ([[pythonFile] [parameters]])

- This folder contains three more python files:
     - common.py - Source file related to camera calibration.     
     - control_exec.py - Python file with most of the developed GUI windows.
     - eval_prec_sec.py - Python file to measure accuracy of the people inside the lab (comparing the real number of people inside the lab and the one meaured by the developed algorithm), along time. It was verified that, for the used bag file, all the entrances/exits of people are reloaded after 16 movements. 

