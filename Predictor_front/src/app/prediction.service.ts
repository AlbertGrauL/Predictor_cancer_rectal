import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface ImageScores {
  polipos: number;
  sangre: number;
  inflamacion: number;
  negativos: number;
}

export interface ClinicalRisk {
  score: number;
  level: 'bajo' | 'moderado' | 'alto' | 'muy_alto';
  active_factors: string[];
}

export interface FusionResult {
  score: number;
  level: 'bajo' | 'moderado' | 'alto' | 'muy_alto';
  image_weight: number;
  clinical_weight: number;
}

export interface PredictionResult {
  // flat (backwards compat)
  polipos: number;
  sangre: number;
  inflamacion: number;
  negativos: number;
  // enriched
  image_scores: ImageScores;
  clinical_risk: ClinicalRisk;
  fusion: FusionResult;
}

@Injectable({ providedIn: 'root' })
export class PredictionService {
  private apiUrl = 'http://localhost:8000/predict';

  constructor(private http: HttpClient) {}

  predict(formData: FormData): Observable<PredictionResult> {
    return this.http.post<PredictionResult>(this.apiUrl, formData);
  }
}
