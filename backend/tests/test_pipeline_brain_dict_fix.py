"""
Test to verify pipeline service correctly handles brain segmentation results with nested dictionaries.
"""
import numpy as np
from unittest.mock import MagicMock, patch


def test_pipeline_handles_brain_results_with_nested_dicts():
    """
    Test that pipeline service correctly filters out nested dictionaries
    from brain segmentation results when saving artifacts.
    
    This addresses the issue where brain_unet returns 'preprocessing' and 'candidates'
    as nested dictionaries that cannot be saved as images.
    """
    # Mock brain segmentation results with both numpy arrays and nested dicts
    mock_brain_results = {
        'mask': np.random.randint(0, 255, (512, 512), dtype=np.uint8),
        'brain_extracted': np.random.randint(0, 255, (512, 512), dtype=np.uint8),
        'overlay': np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8),
        'preprocessing': {  # This is a dict, should be skipped
            'stage1': np.random.randint(0, 255, (512, 512), dtype=np.uint8),
            'stage2': np.random.randint(0, 255, (512, 512), dtype=np.uint8),
        },
        'candidates': {  # This is a dict, should be skipped
            'mask1': np.random.randint(0, 255, (512, 512), dtype=np.uint8),
            'mask2': np.random.randint(0, 255, (512, 512), dtype=np.uint8),
        }
    }
    
    # Track which artifacts were saved
    saved_artifacts = []
    
    def mock_save_artifact(image, image_id, artifact_type):
        # Verify that only numpy arrays are being saved
        assert isinstance(image, np.ndarray), \
            f"save_artifact called with {type(image)}, expected np.ndarray"
        saved_artifacts.append(artifact_type)
        return f"artifacts/{image_id}_{artifact_type}.png"
    
    # Mock the pipeline dependencies
    with patch('app.services.pipeline_service.get_storage_service') as mock_storage:
        with patch('app.services.pipeline_service.get_brain_unet_inference') as mock_brain_unet:
            with patch('app.services.pipeline_service.get_unet_inference') as mock_tumor_unet:
                with patch('app.services.pipeline_service.get_vit_inference') as mock_vit:
                    with patch('app.services.pipeline_service.preprocess_pipeline') as mock_preprocess:
                        # Setup mocks
                        storage_mock = MagicMock()
                        storage_mock.generate_image_id.return_value = "test123"
                        storage_mock.save_upload.return_value = "uploads/test123_original.png"
                        storage_mock.save_artifact.side_effect = mock_save_artifact
                        storage_mock.get_artifact_url.side_effect = lambda x: f"/files/{x}"
                        mock_storage.return_value = storage_mock
                        
                        brain_unet_mock = MagicMock()
                        brain_unet_mock.segment_brain.return_value = mock_brain_results
                        mock_brain_unet.return_value = brain_unet_mock
                        
                        tumor_unet_mock = MagicMock()
                        tumor_unet_mock.segment_image.return_value = {
                            'mask': np.random.randint(0, 255, (512, 512), dtype=np.uint8),
                            'segmented': np.random.randint(0, 255, (512, 512), dtype=np.uint8),
                        }
                        mock_tumor_unet.return_value = tumor_unet_mock
                        
                        vit_mock = MagicMock()
                        vit_mock.classify.return_value = {
                            'class': 'glioma',
                            'confidence': 0.95
                        }
                        mock_vit.return_value = vit_mock
                        
                        mock_preprocess.return_value = {
                            'grayscale': np.random.randint(0, 255, (512, 512), dtype=np.uint8),
                            'denoised': np.random.randint(0, 255, (512, 512), dtype=np.uint8),
                            'normalized': np.random.randint(0, 255, (512, 512), dtype=np.uint8),
                        }
                        
                        # Now test the pipeline
                        from app.services.pipeline_service import PipelineService
                        
                        pipeline = PipelineService()
                        test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
                        
                        # This should NOT raise an error about 'dict' object has no attribute 'shape'
                        result = pipeline.run_inference(test_image)
                        
                        # Verify only numpy arrays were saved for brain segmentation
                        brain_artifacts = [a for a in saved_artifacts if a.startswith('brain_')]
                        
                        # Should have 3 brain artifacts: mask, brain_extracted, overlay
                        # Should NOT have: preprocessing, candidates
                        assert 'brain_mask' in brain_artifacts, "brain_mask should be saved"
                        assert 'brain_brain_extracted' in brain_artifacts, "brain_brain_extracted should be saved"
                        assert 'brain_overlay' in brain_artifacts, "brain_overlay should be saved"
                        assert 'brain_preprocessing' not in brain_artifacts, "brain_preprocessing dict should be skipped"
                        assert 'brain_candidates' not in brain_artifacts, "brain_candidates dict should be skipped"
                        
                        print(f"✓ Test passed: Only {len(brain_artifacts)} brain artifacts saved (dicts skipped)")
                        print(f"  Saved artifacts: {brain_artifacts}")


if __name__ == "__main__":
    test_pipeline_handles_brain_results_with_nested_dicts()
    print("\n✓ Pipeline brain dict fix test passed!")
