import logging
from typing import Optional
# import sitkUtils

import vtk, qt, slicer
import time

from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
)

from slicer import vtkMRMLScalarVolumeNode, vtkMRMLSegmentationNode


try:
    import cv2, json, radiomics, os
    import pickle
    from skimage import filters
    from radiomics import featureextractor
    import SimpleITK as sitk
    import numpy as np
    # from sklearn.preprocessing import StandardScaler
    # from sklearn.svm import SVC


except:
    slicer.util.pip_install('opencv-python')
    slicer.util.pip_install('scikit-image')
    slicer.util.pip_install('scikit-learn')
    slicer.util.pip_install('pyradiomics')
    import cv2, radiomics
    import pickle
    from skimage import filters
    from radiomics import featureextractor
    # from sklearn.preprocessing import StandardScaler
    # from sklearn.svm import SVC
    

#
# semiSegPredict
#


class semiSegPredict(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("semiSegPredict")  # TODO: make this more human readable by adding spaces
        # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Examples")]
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["Binh D. Le (Center for AI in Medical Imaging Research\nChonnam National University, Republic of Korea)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        # _() function marks text as translatable to other languages
        self.parent.helpText = _("""
    An extension for 3D Slicer used to predict whether a urinary stone is pure uric acid
that is a combination of semi-auto segmentation, CT radiomics, and machine learning.
""")
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = _("""
    This extension was an extended work following this scientific article:
Accurate prediction of pure uric acid urinary stones in clinical context via a combination of radiomics and machine learning
(https://doi.org/10.1007/s00345-024-04818-4). 
                                            The work was supported by:
(1) Institute of Information & communications Technology Planning & Evaluation (IITP)
under the Artificial Intelligence Convergence Innovation Human Resources Development (IITP-2023-RS-2023-00256629)
which was funded by the Korea government (MSIT);
(2) Grant from the Ministry of Education, Republic of Korea (NRF-2022R1I1A3072856 and NRF-2021R1I1A3060723);
(3) Grant from Chonnam National University Hospital Biomedical Research Institute (BCRI22037).
                                            """)


#
# semiSegPredictParameterNode
#


@parameterNodeWrapper
class semiSegPredictParameterNode:
    """
    The parameters needed by module.

    inputVolume - The volume to extract radiomics features.
    inputRelativeSeg: The relative segmentation to be semi-auto segmented and radiomics extracted.

    """

    inputVolume: vtkMRMLScalarVolumeNode
    inputRelativeSeg: vtkMRMLSegmentationNode


#
# semiSegPredictWidget
#


class semiSegPredictWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/semiSegPredict.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = semiSegPredictLogic()
        self.logic.showSegmentation = self.addLogSeg
        self.logic.showPrediction = self.addLogPred
        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Buttons
        self.ui.applyButtonSeg.connect("clicked(bool)", self.onApplyButtonSeg)
        self.ui.applyButtonPred.connect("clicked(bool)", self.onApplyButtonPred)

        # Change the display window/level to bone window setting
        volumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
        volumeNode.GetDisplayNode().SetAutoWindowLevel(False)
        volumeNode.GetDisplayNode().SetWindowLevel(1000,400)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """Ensure parameter node exists and observed."""
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.inputVolume:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.inputVolume = firstVolumeNode
        
        if not self._parameterNode.inputRelativeSeg:
            secondVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLSegmentationNode")
            if secondVolumeNode:
                self._parameterNode.inputRelativeSeg = secondVolumeNode

    def setParameterNode(self, inputParameterNode: Optional[semiSegPredictParameterNode]) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
            self._checkCanApply()

    def _checkCanApply(self, caller=None, event=None) -> None:
        if self._parameterNode and self._parameterNode.inputVolume and self._parameterNode.inputRelativeSeg:
            self.ui.applyButtonSeg.toolTip = _("Get Semi-auto Segmentation")
            self.ui.applyButtonSeg.enabled = True
            self.ui.applyButtonPred.toolTip = _("Get Prediction")
            self.ui.applyButtonPred.enabled = True
        else:
            self.ui.applyButtonSeg.toolTip = _("Select input Volume and Relative Segmentation nodes")
            self.ui.applyButtonSeg.enabled = False
            self.ui.applyButtonPred.toolTip = _("Select input Volume and Segmentation nodes")
            self.ui.applyButtonPred.enabled = False

    def onApplyButtonSeg(self) -> None:
        """Run processing when user clicks "Apply" button."""
        with slicer.util.tryWithErrorDisplay(_("Failed to compute results."), waitCursor=True):
            # Compute output
            self.logic.processSeg(self.ui.inputVolume.currentNode(), self.ui.inputRelativeSeg.currentNode())
    
    def onApplyButtonPred(self) -> None:
        """Run processing when user clicks "Apply" button."""
        with slicer.util.tryWithErrorDisplay(_("Failed to compute results."), waitCursor=True):
            # Compute output
            self.logic.processPred(self.ui.inputVolume.currentNode(), self.ui.inputSegPred.currentNode())

    def addLogSeg(self, text):
        """Append text to log window of Semi-auto Segmentation
        """
        self.ui.statusSeg.insertPlainText(text)
        self.ui.statusSeg.insertPlainText("\n")
        self.ui.statusSeg.insertPlainText("\n")

    def addLogPred(self, text):
        """Append text to log window of Prediction
        """
        self.ui.statusPred.insertPlainText(text)
        self.ui.statusPred.insertPlainText("\n")
        self.ui.statusPred.insertPlainText("\n")

#
# semiSegPredictLogic
#


class semiSegPredictLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)
        self.root = os.path.dirname(os.path.realpath(__file__))
        self.TEMP_IMG_DIR = self.root.replace('semiSegPredict',"temp_img.nrrd")
        self.TEMP_SEMI_ROI_DIR = self.root.replace('semiSegPredict',"temp_semiSeg.seg.nrrd")
        self.TEMP_ROI_DIR = self.root.replace('semiSegPredict',"temp_seg.seg.nrrd") 
        self.PARAM_FILE_DIR = self.root.replace('semiSegPredict',"pUA_Params.yaml")
        self.PARAM_FILE_DIR_STONE_INFO = self.root.replace('semiSegPredict',"pUA_Params_no_sampling.yaml")
        self.MODEL_DIR = self.root.replace('semiSegPredict',"model.pkl")
        self.SCALER_DIR = self.root.replace('semiSegPredict',"scaler.pkl") 
        self.showPrediction = None
        self.showSegmentation = None

    def getParameterNode(self):
        return semiSegPredictParameterNode(super().getParameterNode())


    def segmentation (self, inputVolume: vtkMRMLScalarVolumeNode,
                inputRelativeSeg: vtkMRMLSegmentationNode):
        """
        Semi-auto segmentation the Relative ROI by image processing
        """
        imageArray = slicer.util.arrayFromVolume(inputVolume)
        
        relSegmentation = inputRelativeSeg.GetSegmentation()
        relSegment = relSegmentation.GetNthSegment(0)
        relSegmentID = relSegmentation.GetSegmentIdBySegment(relSegment)
        
        roiArray = slicer.util.arrayFromSegmentBinaryLabelmap(inputRelativeSeg, relSegmentID, referenceVolumeNode=inputVolume)

        stone = np.multiply(imageArray, roiArray)

        slice_index = []
        for i in range (len(stone)):
            if np.any(stone[i]):
                slice_index.append(i)
            
        min_mean_HU_rim = 10000
        for j in slice_index:
            stone_2 = stone[j]
            stone_2 = np.where(stone_2 < 0, 0, stone_2)
            # Edge detection using Scharr operator
            edge_scharr = filters.scharr(stone_2)
            # Take the 98th percentile of the gradient to be the threshold
            limit = np.percentile(edge_scharr[edge_scharr!=0].ravel(), [98])[0]            
            # Create binary mask of the edge
            rim_mask = np.where(edge_scharr < limit, 0, 1)
            # assume soft tissue < 80HU
            rim = stone_2 * rim_mask
            rim = np.where(rim < 80, 0, rim)
            rim_mask_2 = np.where(rim > 0, 1, 0)
            # Calculate the mean HU of the edge
            sum_HU_rim = rim.sum()
            num_pixel_rim = rim_mask_2.sum()
            # avoid error in tiny part of stone presence
            if num_pixel_rim == 0:
                mean_HU_rim = 0
            else:
                mean_HU_rim = sum_HU_rim/num_pixel_rim   
            if mean_HU_rim < min_mean_HU_rim:
                min_mean_HU_rim = mean_HU_rim
            # assume soft tissue < 70HU

        if min_mean_HU_rim < 80:
            min_mean_HU_rim = 80
        elif min_mean_HU_rim > 150:
            min_mean_HU_rim = 150        

        for j in slice_index:
            stone_2 = stone[j]
            # Define any pixel > the value belongs the to the stones
            stone_3 =  np.where(stone_2 > min_mean_HU_rim, 1, 0).astype('int16')
            # Dilate to expand the stone a bit outside of the stone
            stone_3 = cv2.erode (stone_3, np.ones((2,1), np.uint8), iterations=1)
            stone_3 = cv2.dilate(stone_3, np.ones((2,1), np.uint8), anchor=(0, 0), iterations=1)
            stone_3 = cv2.erode (stone_3, np.ones((1,2), np.uint8), iterations=1)            
            stone_3 = cv2.dilate(stone_3, np.ones((2,2), np.uint8), anchor=(0, 0), iterations=1)
            # Erosion and Dilation one more time
            stone_3 = cv2.erode (stone_3, np.ones((2,2), np.uint8), anchor=(0, 0), iterations=1)
            stone_3 = cv2.dilate(stone_3, np.ones((2,2), np.uint8), iterations=1)
            stone_3 = cv2.dilate(stone_3, np.ones((2,2), np.uint8), anchor=(1, 1), iterations=1)
            # Make sure the corrected stone is within the initial mask
            new_stone = np.multiply(stone_3, roiArray[j])
            # Replace the slide to corresponding of the stone
            stone[j] = new_stone
        
        slicer.util.exportNode(inputVolume, self.TEMP_IMG_DIR)
        imageITK = sitk.ReadImage(self.TEMP_IMG_DIR)

        new_mask = sitk.GetImageFromArray(stone)
        new_mask.CopyInformation(imageITK)
        sitk.WriteImage(new_mask, self.TEMP_SEMI_ROI_DIR)
        semiSegName = self.displaySemiAutoSegmentation()
        imgName = inputVolume.GetName()
        relSegName = inputRelativeSeg.GetName()
        return semiSegName, imgName, relSegName

    def displaySemiAutoSegmentation(self):
        """
        Show semi-auto segmentation on the 3D Slicer viewers
        """
        
        scene = slicer.mrmlScene
        # Get all vtkMRMLSegmentationNode objects
        segmentation_nodes = scene.GetNodesByClass("vtkMRMLSegmentationNode")
        # Iterate through the nodes and print their names
        number_of_seg = segmentation_nodes.GetNumberOfItems()

        labelmapVolumeNode = slicer.util.loadLabelVolume(self.TEMP_SEMI_ROI_DIR)
        segNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode",
                                                     f'Semi-auto_ROI_{number_of_seg}')
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labelmapVolumeNode, segNode)
        slicer.mrmlScene.RemoveNode(labelmapVolumeNode)

        segmentation = segNode.GetSegmentation()
        segment = segmentation.GetNthSegment(0)
        segmentID = segmentation.GetSegmentIdBySegment(segment)
        # Set the color of the segment to a random RGB value
        R, G, B = np.random.randint(0, 255, 3)
        segNode.GetSegmentation().GetSegment(segmentID).SetColor(R, G, B)

        return segNode.GetName()
        
    def radiomicsExtraction (self):
        """
        Extract radiomics features from image and semi-auto segmented ROI
        """
        logger = radiomics.logging.getLogger("radiomics")
        logger.setLevel(radiomics.logging.ERROR)
        extractor = featureextractor.RadiomicsFeatureExtractor(self.PARAM_FILE_DIR)
        radiomics_fts = extractor.execute (self.TEMP_IMG_DIR,
                                           self.TEMP_ROI_DIR)
        input_sample = []
        for key, value in radiomics_fts.items():
            if key.startswith('original') or key.startswith('wavelet'):
                input_sample.append(float(value))
        return input_sample
    
    def get_train_val_id_from_fold (self, fold, data):
        """
        Get sample IDs in validation and training subsets of the fold
        """
        train_ids, val_ids =[], []
        for sample_id, v in data.items():
            if v['fold_'+ str(fold)] == 'train':
                train_ids.append(sample_id)
            if v['fold_'+ str(fold)] == 'val':
                val_ids.append(sample_id)
        return train_ids, val_ids

    def get_radiomics_label(self, ids, data):
        """
        Get radiomics features and labels of sample with IDs are ids
        """
        features, label = [],[]
        for i in ids:
            label.append(data[i]['label'])
            features.append(list(data[i]['radiomics'].values()))
        features = np.asanyarray(features)
        return features, label

    def get_stone_info (self):
        extractor = featureextractor.RadiomicsFeatureExtractor(self.PARAM_FILE_DIR_STONE_INFO)
        radiomics_fts = extractor.execute (self.TEMP_IMG_DIR,
                                           self.TEMP_ROI_DIR)
        stone_info = {}
        for key, value in radiomics_fts.items():
            if key in ['original_shape_MeshVolume',
                       'original_firstorder_Mean',
                       'original_firstorder_Variance',
                       'original_shape_Maximum2DDiameterSlice',
                       'original_firstorder_Maximum'
                       ]:
                stone_info[key]=float(value)
        return stone_info

    def getPrediction(self, input_sample):
        """
        Get prediction from the pretrained pUA prediction model
        """
        features_selected_index = [20, 28, 214, 233, 323, 330, 420, 544, 605, 395] ### current set of features
        
        model = pickle.load(open(self.MODEL_DIR, 'rb'))
        scaler = pickle.load(open(self.SCALER_DIR, 'rb'))

        input_sample = np.array([input_sample])
        input_sample  = scaler.transform(input_sample)
        input_sample  = input_sample[:, features_selected_index]

        predicted_label = model.predict(input_sample)
        predicted_prob = model.predict_proba(input_sample)
        
        if predicted_label == -1:
            label = 'Non-UA'
            prob = round(predicted_prob[0][0]*100, 2)
        else:
            label = 'pUA'
            prob = round(predicted_prob[0][1]*100, 2)
        return label, prob

    def processSeg(self,
                inputVolume: vtkMRMLScalarVolumeNode,
                inputRelativeSeg: vtkMRMLSegmentationNode):
        """
        Run the Semi-auto Segmentation algorithm.
        Can be used without GUI widget.
        params:
            inputVolume: volume to be segmented
            inputRelativeSeg: corresponding relative segmentation
        --> show output volume in slice viewers
        """

        if not inputVolume or not inputRelativeSeg:
            raise ValueError("Input or output volume is invalid")
        logging.info("Processing started")

        # Semi-auto segmentation task
        startTimeSeg = time.time()
        logging.info("Semi-auto segmentation started")

        semiSegName, imgName, relSegName = self.segmentation(inputVolume, inputRelativeSeg)
        lm = slicer.app.layoutManager()
        lm.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUpRedSliceView)

        stopTimeSeg = time.time()
        logging.info(f"Processing completed in {stopTimeSeg-startTimeSeg:.2f} seconds")

        self.showSegmentation(f"The ROI: [{semiSegName}]\
                              \nwas generated from the Images: [{imgName}] and the Relative ROI: [{relSegName}]\
                \n\nNOTE: Please check the result\
                \n(Processing time: [{stopTimeSeg-startTimeSeg:.2f}] seconds)\n\n\n\n")


    def exportLabelmap(self,segmentationNode, referenceVolumeNode, filepath):
        labelmapVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
        slicer.modules.segmentations.logic().ExportVisibleSegmentsToLabelmapNode(segmentationNode, labelmapVolumeNode, referenceVolumeNode)
        slicer.util.saveNode(labelmapVolumeNode, filepath)
        slicer.mrmlScene.RemoveNode(labelmapVolumeNode.GetDisplayNode().GetColorNode())
        slicer.mrmlScene.RemoveNode(labelmapVolumeNode)

    def processPred(self,
                    inputVolume: vtkMRMLScalarVolumeNode,
                    inputSeg: vtkMRMLSegmentationNode):
        """
        Run radiomics features extraction and get prediction.
        Can be used without GUI widget.
        params:
            inputVolume: volume to be segmented
            inputSeg: corresponding segmentation, could be the semi-auto segmentation
        --> show prediction result in the log window
        """
        volName = inputVolume.GetName()
        segName = inputSeg.GetName()
        self.exportLabelmap(inputSeg, inputVolume, self.TEMP_ROI_DIR)

        # Prediction task
        startTimePred = time.time()
        logging.info("Prediction started")

        input_sample = self.radiomicsExtraction()
        label, prob = self.getPrediction(input_sample)

        stone_info = self.get_stone_info() #############
        stone_volume = round(stone_info['original_shape_MeshVolume'])
        max_dia_slice = round(stone_info['original_shape_Maximum2DDiameterSlice'], 1)
        max_HU = round(stone_info['original_firstorder_Maximum'])
        mean_HU = round(stone_info['original_firstorder_Mean'])
        standard_deviation = round(stone_info['original_firstorder_Variance'] ** 0.5)

        stopTimePred = time.time()
        logging.info(f"Processing completed in {stopTimePred-startTimePred:.2f} seconds")

        self.showPrediction(f"This was predicted as a: [{label}] stone with probability: [{prob}%]\
                            \nInput: the Images: [{volName}] and the ROI: [{segName}]\
                            \n\nDISCLAIMER: Use at your own risk\
                            \n(Processing time: [{stopTimePred-startTimePred:.2f}] seconds)\
                            \n\nSTONE INFORMATION: Volume: [{stone_volume}]; Max axial diameter: [{max_dia_slice}];\
                            \nMax HU: [{max_HU}]; Mean HU ± SD: [{mean_HU} ± {standard_deviation}]\n\n\n\n"
                            )

#
# semiSegPredictTest
#


class semiSegPredictTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_semiSegPredict1()

    def test_semiSegPredict1(self):
        """Add test here later."""
        self.delayDisplay("Starting the test")
        self.delayDisplay("Test passed!")
