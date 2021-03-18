"""
unit_test.py is designed to for testing the Fall Detection System software and not apart of the actual System's code base.


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
import cv2
import unittest
import psycopg2
import ComputerVision
import Templates
import HumanStateClassifier

class TestFallCasesHelper:
    def displayTestCV(self, local, fileName):
        templates = self.loadTemplates(local=local)

        # Classifiers
        edge_classifier = HumanStateClassifier.KNeighborsClassifier(templates['edge'], k=4)
        foreground_classifier = HumanStateClassifier.KNeighborsClassifier(templates['foreground'], k=4)

        # Computer Vision
        return ComputerVision.display(foregroundClassifier=foreground_classifier,
                                    edgeClassifier=edge_classifier,
                                    videoPath=fileName,
                                    saveTemplate=False,
                                    checkTemplate=True)

    def loadTemplates(self, local):
        if (local == True):
            # Load Templates Locally
            templates = Templates.loadTemplatesLocally()
        else:
            # Load Templates from Database
            LOCAL_DATABASE_NAME = "postgres"
            LOCAL_DATABASE_PASSWORD = "password"
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

        return templates

# Basic Framework For Unit Testing Fall Detection

class TestFallCases(unittest.TestCase):
    def test0_5_female(self):
        test_case_helper = TestFallCasesHelper()
        fileName = './fall_samples/fall-0-5-1.mp4'
        fall_detected = test_case_helper.displayTestCV(local=True, fileName=fileName)
        self.assertEqual(fall_detected, True)

    def test0_5_foreward_fall(self):
        test_case_helper = TestFallCasesHelper()
        fileName = './fall_samples/fall-0-5-2.mp4'
        fall_detected = test_case_helper.displayTestCV(local=True, fileName=fileName)
        self.assertEqual(fall_detected, True)
    
    def test5_10_forewards(self):
        test_case_helper = TestFallCasesHelper()
        fileName = './fall_samples/fall-5-10-1.mp4'
        fall_detected = test_case_helper.displayTestCV(local=True, fileName=fileName)
        self.assertEqual(fall_detected, True)
    
    def test15_20_1(self):
        test_case_helper = TestFallCasesHelper()
        fileName = './fall_samples/fall-15-20-1.mp4'
        fall_detected = test_case_helper.displayTestCV(local=True, fileName=fileName)
        self.assertEqual(fall_detected, True)

    def test15_20_2(self):
        test_case_helper = TestFallCasesHelper()
        fileName = './fall_samples/fall-15-20-2.mp4'
        fall_detected = test_case_helper.displayTestCV(local=True, fileName=fileName)
        self.assertEqual(fall_detected, True)

    def testLowLight_15_20(self):
        test_case_helper = TestFallCasesHelper()
        fileName = './fall_samples/fall-lowlight.mp4'
        fall_detected = test_case_helper.displayTestCV(local=True, fileName=fileName)
        self.assertEqual(fall_detected, True)
    
    def testObstructed_15_20(self):
        test_case_helper = TestFallCasesHelper()
        fileName = './fall_samples/fall-obstructed.mp4'
        fall_detected = test_case_helper.displayTestCV(local=True, fileName=fileName)
        self.assertEqual(fall_detected, True)

    def testDetectHuman(self):
        test_case_helper = TestFallCasesHelper()
        fileName = './fall_samples/dogs.mp4'
        fall_detected = False
        fall_detected = test_case_helper.displayTestCV(local=True, fileName=fileName)
        self.assertEqual(fall_detected, False)

    def testSittingDown(self):
        test_case_helper = TestFallCasesHelper()
        fileName = './fall_samples/human-sitting-down.mp4'
        fall_detected = False
        fall_detected = test_case_helper.displayTestCV(local=True, fileName=fileName)
        self.assertEqual(fall_detected, False)

    def testLyingDown(self):
        test_case_helper = TestFallCasesHelper()
        fileName = './fall_samples/human-lying-down.mp4'
        fall_detected = False
        fall_detected = test_case_helper.displayTestCV(local=True, fileName=fileName)
        self.assertEqual(fall_detected, False)

    def testRemoveNonMovingEntites(self):
        test_case_helper = TestFallCasesHelper()
        fileName = './fall_samples/static-with-window.mp4'
        test_case_helper.displayTestCV(local=True, fileName=fileName)
    
    def testCamera(self):
        test = True
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            test = False
        self.assertEqual(test,True)
    
    def testDBConnection(self):
        try:
            self.conn = psycopg2.connect(host = 'localhost', \
                database = 'CSC-450_FDS', user = 'postgres', \
                password = 'Apcid28;6jdn')
            test = True
        except (Exception, psycopg2.DatabaseError):
            test = False
        self.assertEqual(test, True)

if __name__ == '__main__':
    unittest.main()
