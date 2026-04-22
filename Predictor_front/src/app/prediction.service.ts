import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface PredictionResult {
  polipos: number;
  sangre: number;
  inflamacion: number;
  negativos: number;
}

@Injectable({ providedIn: 'root' })
export class PredictionService {
  private apiUrl = 'http://localhost:8000/predict';

  constructor(private http: HttpClient) {}

  predict(formData: FormData): Observable<PredictionResult> {
    return this.http.post<PredictionResult>(this.apiUrl, formData);
  }
}
