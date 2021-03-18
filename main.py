"""
main.py is the launch file for the Fall Detection System.



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

import UserInterface
import TemplateViewer
import Templates
import HumanStateClassifier
import ComputerVision

def main():
    print('Starting FDSystem')

    # Database credentials. 
    LOCAL_DATABASE_NAME = 'CSC-450_FDS'
    LOCAL_DATABASE_PASSWORD = 'Pd9;$chsi9-$nc'

    database = Templates.TemplateDatabase(LOCAL_DATABASE_NAME, LOCAL_DATABASE_PASSWORD)
    database.connect()

    templates = {'edge': {}, 'foreground': {}}
    template_characteristics = templates.keys()
    template_types = ['upright', 'falling', 'sitting', 'lying']

    for template_characteristic in template_characteristics:
        for template_type in template_types:
            templates[template_characteristic][template_type] = []

    if database.connected():
        templates = database.load_templates(templates)
    else:
        templates = Templates.loadTemplatesLocally()

    edge_classifier = HumanStateClassifier.KNeighborsClassifier(templates['edge'], k=20)
    foreground_classifier = HumanStateClassifier.KNeighborsClassifier(templates["foreground"], k=20)

    UserInterface.systemUserInterface(templates,
                                    database,
                                    edgeClassifier=edge_classifier,
                                    foregroundClassifier=foreground_classifier)

if __name__ == '__main__':
    main()