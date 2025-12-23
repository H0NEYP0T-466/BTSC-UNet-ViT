/**
 * API client for BTSC-UNet-ViT backend.
 */
import axios from 'axios';
import type { AxiosInstance } from 'axios';
import type {
  InferenceResponse,
  PreprocessResponse,
  SegmentResponse,
  ClassifyResponse,
  HealthResponse
} from './types';

const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8080';

class ApiClient {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: BASE_URL,
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    // Log requests for debugging
    this.client.interceptors.request.use((config) => {
      console.log(`[API] ${config.method?.toUpperCase()} ${config.url}`);
      return config;
    });

    // Log responses
    this.client.interceptors.response.use(
      (response) => {
        console.log(`[API] Response from ${response.config.url}:`, response.data);
        return response;
      },
      (error) => {
        console.error('[API] Error:', error.response?.data || error.message);
        return Promise.reject(error);
      }
    );
  }

  /**
   * Check API health status.
   */
  async healthCheck(): Promise<HealthResponse> {
    const response = await this.client.get<HealthResponse>('/api/health');
    return response.data;
  }

  /**
   * Preprocess an image.
   */
  async preprocessImage(file: File): Promise<PreprocessResponse> {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await this.client.post<PreprocessResponse>('/api/preprocess', formData);
    return response.data;
  }

  /**
   * Segment an image using UNet.
   */
  async segmentImage(file: File): Promise<SegmentResponse> {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await this.client.post<SegmentResponse>('/api/segment', formData);
    return response.data;
  }

  /**
   * Classify an image using ViT.
   */
  async classifyImage(file: File): Promise<ClassifyResponse> {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await this.client.post<ClassifyResponse>('/api/classify', formData);
    return response.data;
  }

  /**
   * Run full inference pipeline.
   */
  async runInference(file: File, skipPreprocessing: boolean = false): Promise<InferenceResponse> {
    console.log('[API] Starting full inference pipeline, skipPreprocessing:', skipPreprocessing);
    const formData = new FormData();
    formData.append('file', file);
    formData.append('skip_preprocessing', String(skipPreprocessing));
    
    const response = await this.client.post<InferenceResponse>('/api/inference', formData);
    console.log('[API] Inference completed:', response.data);
    return response.data;
  }

  /**
   * Get full URL for a resource.
   */
  getResourceUrl(path: string): string {
    if (path.startsWith('http')) {
      return path;
    }
    return `${BASE_URL}${path}`;
  }
}

export const apiClient = new ApiClient();
