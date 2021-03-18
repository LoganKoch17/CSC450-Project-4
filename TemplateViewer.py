"""
TemplateViewer.py is apart of the Templates component referenced in section 3.2 of the SDD, A03_SDD_Team4.docx. 




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
from cv2 import cv2
import numpy as np

"""
TemplateViewer referenced and defined in section 2.0 of the SDD, A03_SDD_Team4.docx. 
"""
class TemplateViewer:
    template_type = 'upright'
    template_characteristic = 'edge'

    def __init__(self, templates):
        self.templates = templates
        self.current_image = None
        self.images = np.array([])
        self.images_length = len(self.images)
        self.current_image_index = 0

    # UI functionality to view templates.
    def view_templates(self):

        self.update_image_list(self.template_characteristic, self.template_type)

        print("""
                Use following commands to sort viewing template.
                Command:   Template Type:
                1----------Upright
                2----------Falling
                3----------Sitting Down
                4----------Lying Down

                           Template Characteristic:
                f----------foreground
                e----------edge  
                """)

        while True:

            cv2.imshow("current_image", self.current_image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            elif key == ord('['):
                if 0 < self.current_image_index:
                    self.current_image_index -= 1
                print(self.current_image_index)
                self.current_image = self.images[self.current_image_index]

            elif key == ord(']'):
                if self.images_length - 1 > self.current_image_index:
                    self.current_image_index += 1
                print(self.current_image_index)
                self.current_image = self.images[self.current_image_index]
            
            elif key == ord('e'):
                self.current_image_index = 0
                self.update_image_list('edge', self.template_type)

            elif key == ord('f'):
                self.current_image_index = 0
                self.update_image_list('foreground', self.template_type)
            
            elif key == ord('1'):
                self.current_image_index = 0
                self.current_image_index = 0
                self.update_image_list(self.template_characteristic, 'upright')

            elif key == ord('2'):
                self.current_image_index = 0
                self.current_image_index = 0
                self.update_image_list(self.template_characteristic, 'falling')
                
            elif key == ord('3'):
                self.current_image_index = 0
                self.current_image_index = 0
                self.update_image_list(self.template_characteristic, 'sitting')

            elif key == ord('4'):
                self.current_image_index = 0
                self.current_image_index = 0
                self.update_image_list(self.template_characteristic, 'lying')
                
        cv2.destroyAllWindows()

    def update_image_list(self, templateCharacteristic, templateType):
        print(f"templateCharacteristic: {templateCharacteristic}\nType: {templateType}")
        self.template_characteristic = templateCharacteristic
        self.template_type = templateType
        self.images = np.array(self.templates[self.template_characteristic][self.template_type])
        self.images_length = len(self.images)
        self.current_image = self.images[self.current_image_index]