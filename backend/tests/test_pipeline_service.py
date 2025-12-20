"""
Test for the new pipeline flow: preprocessing -> classification -> conditional segmentation
"""
import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.pipeline_service import PipelineService


class TestPipelineService(unittest.TestCase):
    """Test the updated pipeline service with conditional segmentation."""
    
    def setUp(self):
        """Setup test fixtures."""
        self.pipeline = PipelineService()
        
    @patch('app.services.pipeline_service.preprocess_pipeline')
    @patch('app.services.pipeline_service.get_storage_service')
    def test_pipeline_skips_segmentation_for_notumor(self, mock_storage_service, mock_preprocess):
        """Test that segmentation is skipped when ViT classifies as notumor."""
        # Setup mocks
        mock_storage = Mock()
        mock_storage.generate_image_id.return_value = 'test-image-id'
        mock_storage.save_upload.return_value = 'upload-url'
        mock_storage.save_artifact.return_value = 'artifact-url'
        mock_storage.get_artifact_url.return_value = 'http://example.com/artifact'
        mock_storage_service.return_value = mock_storage
        
        # Mock preprocessing
        mock_preprocess.return_value = {
            'normalized': np.random.rand(256, 256).astype(np.float32)
        }
        
        # Mock ViT to return notumor
        self.pipeline.vit = Mock()
        self.pipeline.vit.classify.return_value = {
            'class': 'notumor',
            'confidence': 0.95,
            'logits': [0.1, 0.05, 0.05, 0.8],
            'probabilities': [0.1, 0.05, 0.05, 0.8]
        }
        
        # Mock UNet (should not be called)
        self.pipeline.tumor_unet = Mock()
        
        # Create test image
        test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        # Run inference
        result = self.pipeline.run_inference(test_image)
        
        # Assertions
        self.assertEqual(result['classification']['class'], 'notumor')
        self.assertNotIn('tumor_segmentation', result)  # Should NOT have segmentation
        
        # Verify UNet was never called
        self.pipeline.tumor_unet.segment_image.assert_not_called()
        
        print("✅ Test passed: Segmentation skipped for notumor classification")
        
    @patch('app.services.pipeline_service.preprocess_pipeline')
    @patch('app.services.pipeline_service.get_storage_service')
    def test_pipeline_runs_segmentation_for_tumor(self, mock_storage_service, mock_preprocess):
        """Test that segmentation runs when ViT classifies as tumor."""
        # Setup mocks
        mock_storage = Mock()
        mock_storage.generate_image_id.return_value = 'test-image-id'
        mock_storage.save_upload.return_value = 'upload-url'
        mock_storage.save_artifact.return_value = 'artifact-url'
        mock_storage.get_artifact_url.return_value = 'http://example.com/artifact'
        mock_storage_service.return_value = mock_storage
        
        # Mock preprocessing
        mock_preprocess.return_value = {
            'normalized': np.random.rand(256, 256).astype(np.float32)
        }
        
        # Mock ViT to return glioma
        self.pipeline.vit = Mock()
        self.pipeline.vit.classify.return_value = {
            'class': 'glioma',
            'confidence': 0.92,
            'logits': [0.8, 0.1, 0.05, 0.05],
            'probabilities': [0.8, 0.1, 0.05, 0.05]
        }
        
        # Mock UNet
        self.pipeline.tumor_unet = Mock()
        self.pipeline.tumor_unet.segment_image.return_value = {
            'segmented': np.random.rand(256, 256).astype(np.float32),
            'mask': np.random.randint(0, 2, (256, 256), dtype=np.uint8)
        }
        
        # Create test image
        test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        # Run inference
        result = self.pipeline.run_inference(test_image)
        
        # Assertions
        self.assertEqual(result['classification']['class'], 'glioma')
        self.assertIn('tumor_segmentation', result)  # SHOULD have segmentation
        
        # Verify UNet was called
        self.pipeline.tumor_unet.segment_image.assert_called_once()
        
        print("✅ Test passed: Segmentation runs for tumor classification")
        
    @patch('app.services.pipeline_service.preprocess_pipeline')
    @patch('app.services.pipeline_service.get_storage_service')
    def test_pipeline_runs_segmentation_for_all_tumor_types(self, mock_storage_service, mock_preprocess):
        """Test that segmentation runs for all tumor types."""
        tumor_types = ['glioma', 'meningioma', 'pituitary']
        
        for tumor_type in tumor_types:
            with self.subTest(tumor_type=tumor_type):
                # Setup mocks
                mock_storage = Mock()
                mock_storage.generate_image_id.return_value = 'test-image-id'
                mock_storage.save_upload.return_value = 'upload-url'
                mock_storage.save_artifact.return_value = 'artifact-url'
                mock_storage.get_artifact_url.return_value = 'http://example.com/artifact'
                mock_storage_service.return_value = mock_storage
                
                # Mock preprocessing
                mock_preprocess.return_value = {
                    'normalized': np.random.rand(256, 256).astype(np.float32)
                }
                
                # Mock ViT to return tumor type
                self.pipeline.vit = Mock()
                self.pipeline.vit.classify.return_value = {
                    'class': tumor_type,
                    'confidence': 0.90,
                    'logits': [0.7, 0.1, 0.1, 0.1],
                    'probabilities': [0.7, 0.1, 0.1, 0.1]
                }
                
                # Mock UNet
                self.pipeline.tumor_unet = Mock()
                self.pipeline.tumor_unet.segment_image.return_value = {
                    'segmented': np.random.rand(256, 256).astype(np.float32),
                    'mask': np.random.randint(0, 2, (256, 256), dtype=np.uint8)
                }
                
                # Create test image
                test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
                
                # Run inference
                result = self.pipeline.run_inference(test_image)
                
                # Assertions
                self.assertEqual(result['classification']['class'], tumor_type)
                self.assertIn('tumor_segmentation', result)
                
                print(f"✅ Test passed: Segmentation runs for {tumor_type}")


class TestPipelineFlow(unittest.TestCase):
    """Test the order of operations in the pipeline."""
    
    @patch('app.services.pipeline_service.preprocess_pipeline')
    @patch('app.services.pipeline_service.get_storage_service')
    def test_classification_before_segmentation(self, mock_storage_service, mock_preprocess):
        """Test that classification happens before segmentation."""
        pipeline = PipelineService()
        
        # Track call order
        call_order = []
        
        # Setup mocks
        mock_storage = Mock()
        mock_storage.generate_image_id.return_value = 'test-image-id'
        mock_storage.save_upload.return_value = 'upload-url'
        mock_storage.save_artifact.return_value = 'artifact-url'
        mock_storage.get_artifact_url.return_value = 'http://example.com/artifact'
        mock_storage_service.return_value = mock_storage
        
        mock_preprocess.return_value = {
            'normalized': np.random.rand(256, 256).astype(np.float32)
        }
        
        # Mock ViT
        def vit_classify_side_effect(*args, **kwargs):
            call_order.append('vit_classify')
            return {
                'class': 'glioma',
                'confidence': 0.90,
                'logits': [0.7, 0.1, 0.1, 0.1],
                'probabilities': [0.7, 0.1, 0.1, 0.1]
            }
        
        pipeline.vit = Mock()
        pipeline.vit.classify.side_effect = vit_classify_side_effect
        
        # Mock UNet
        def unet_segment_side_effect(*args, **kwargs):
            call_order.append('unet_segment')
            return {
                'segmented': np.random.rand(256, 256).astype(np.float32),
                'mask': np.random.randint(0, 2, (256, 256), dtype=np.uint8)
            }
        
        pipeline.tumor_unet = Mock()
        pipeline.tumor_unet.segment_image.side_effect = unet_segment_side_effect
        
        # Create test image
        test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        # Run inference
        result = pipeline.run_inference(test_image)
        
        # Verify order
        self.assertEqual(call_order, ['vit_classify', 'unet_segment'])
        print("✅ Test passed: Classification happens before segmentation")


if __name__ == '__main__':
    print("=" * 80)
    print("Testing Pipeline Service - New Flow")
    print("=" * 80)
    print("\nRunning tests...")
    unittest.main(verbosity=2)
