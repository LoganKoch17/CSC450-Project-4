"""
UserInterface.py is the User Interface component referenced in section 3.2 of the SDD, A03_SDD_Team4.docx. 



Copyright (c) 2020 Fall Detection System, All rights reserved.
Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1.	Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2.	Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer
    in the documentation and/or other materials provided with the distribution.

3.	Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived
    from this software without specific prior written permission. 

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

"""
import ComputerVision
import TemplateViewer
import Templates


# Creates the user interface in the console.
def systemUserInterface(templates, database, foregroundClassifier, edgeClassifier):

    while True:

        print("""
        Command:(button)                      Description:
        Video:(1)                             Displays available videos in 'fall_samples' with computer vision.
        Webcam:(2)                            Displays connected webcam with computer vision
        View Templates:(3)                    Allows user to modify templates that exist in the database.
        Make Singular Comparison:(4)          Demonstrates comparing a template to a frame.
        Access Database:(5)                   Access Database UI.
        Load Templates Locally:(6)            Loads templates from files manually instead of the database. 
        QUIT(q)
        """)

        command = input("Command: ")

        if command == '1':

            available_videos = ['./fall_samples/fall-01-cam0.mp4', './fall_samples/fall-27-cam0.mp4', './fall_samples/fall-obstructed.mp4', './fall_samples/human-sitting-down.mp4', './fall_samples/fall-0-5-2.mp4', './fall_samples/fall-5-10-1.mp4', './fall_samples/fall-15-20-1.mp4','./fall_samples/fall-15-20-2.mp4', './fall_samples/fall-lowlight.mp4', './fall_samples/fall-obstructed.mp4']

            print("Please choose from: ")

            i = 0
            for video in available_videos:
                print(str(i) + video)
                i+=1

            try:
                selection = int(input("Enter: "))

                if selection < i:

                    print("Would you like to save the frames as templates?(y/n):")
                    save_templates = input()
                    save_templates = True if save_templates == 'y' else False

                    print("Would you like to compare templates to video frame?(y/n):")
                    check_templates = input()
                    check_templates = True if check_templates == 'y' else False

                    ComputerVision.display(foregroundClassifier=foregroundClassifier,
                                        edgeClassifier=edgeClassifier,
                                        videoPath = available_videos[selection], 
                                        saveTemplate = save_templates, 
                                        checkTemplate = check_templates)
                else:

                    print("Incorrect selection.")

            except ValueError as error:

                print(error)

        elif command == '2':

            session_name = input("Session Name (leave blank for default session name): ")

            if session_name == "":

                ComputerVision.display(foregroundClassifier=foregroundClassifier,
                                    edgeClassifier=edgeClassifier,
                                    saveTemplate=False,
                                    checkTemplate=False)
            else:

                ComputerVision.display(foregroundClassifier=foregroundClassifier,
                                    edgeClassifier=edgeClassifier,
                                    saveTemplate=True,
                                     checkTemplate=False,
                                      sessionName= session_name)
                                      
        elif command == '3':

            template_modifier = TemplateViewer.TemplateViewer(templates)
            template_modifier.view_templates()

        elif command == '4': 
            
            comparison_frame = templates['edge']['upright'][0]
            ComputerVision.showImage(comparison_frame)

            edge_classification = edgeClassifier.classify(comparison_frame)
            print(f'edge_classification: {edge_classification}')

            foreground_classification = foregroundClassifier.classify(comparison_frame)
            print(f'foreground_classification: {foreground_classification}')

        elif command == '5':

            if database.connected():
                databaseUserInterface(database)
                templates = database.load_templates(templates)

            else:
                print("Database not connected.")

        elif command == '6':
            templates = Templates.loadTemplatesLocally()

        elif command == 'q':
            break

        else:
            print("incorrect command.")

# User Interface functionality relating to the database.
def databaseUserInterface(database):

    database.connect()

    show_UI = database.connected()

    while show_UI:

        print("""
        Command:                        Description:\n
        1                               Add template.\n
        2                               Delete template.\n
        3                               Access template image by template_id.\n
        4                               Upload all templates locally.\n
        5                               Delete all entries.\n
        Previous Menu(r)                Returns to preivous menu.
        """)

        command = input("Enter a command: ")
        
        if command == "1":

            image_path = input("Please enter the path of the image: ")
            template_type = input("Please enter the template_type: ")
            template_characteristic = input("Please enter template_characteristic: ")
            
            image_byte_array = ComputerVision.imagePathToByteArray(image_path)

            image_name = image_path.split('/')
            image_name = image_name[-1]

            database.add_template(template_type, template_characteristic, image_name, image_byte_array)

        elif command == "2":

            template_id = input("Enter the template_id you wish to delete: ")
            database.delete_template(template_id)

        elif command == "3":

            template_id = input("Enter template_id: ")
            image = database.access_image_by_id(template_id)
            ComputerVision.showImage(image)

        elif command == "4":

            database.upload_all_local_templates()
        
        elif command == "5":

            id_list = database.list_of_all_IDs()

            for template_id in id_list:
                database.delete_template(template_id)

        elif command == "r":

            show_UI = False

        else:
            print("\nINCORRECT COMMAND ENTERED")
