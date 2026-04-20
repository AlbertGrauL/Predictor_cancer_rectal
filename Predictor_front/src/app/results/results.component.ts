import { Component } from '@angular/core';
import { CommonModule, PercentPipe } from '@angular/common';
import { RouterLink, Router } from '@angular/router';
import { PredictionResult } from '../prediction.service';

interface CategoryEntry {
  key: keyof PredictionResult;
  label: string;
  prob: number;
}

@Component({
  selector: 'app-results',
  standalone: true,
  imports: [RouterLink, CommonModule, PercentPipe],
  templateUrl: './results.component.html',
  styleUrl: './results.component.css'
})
export class ResultsComponent {
  result: PredictionResult | null = null;
  categories: CategoryEntry[] = [];
  topCategory: CategoryEntry | null = null;
  imageDataUrl: string | null = null;

  private labelMap: Record<keyof PredictionResult, string> = {
    polipos: 'Pólipos',
    sangre: 'Sangre',
    inflamacion: 'Inflamación',
    negativos: 'Negativo'
  };

  constructor(private router: Router) {
    const nav = this.router.getCurrentNavigation();
    const state = nav?.extras?.state;
    const result = state?.['result'] as PredictionResult | undefined;

    if (!result) {
      this.router.navigate(['/upload']);
      return;
    }

    this.result = result;
    this.imageDataUrl = state?.['imageDataUrl'] ?? null;
    this.categories = (Object.keys(result) as (keyof PredictionResult)[]).map(k => ({
      key: k,
      label: this.labelMap[k],
      prob: result[k]
    }));
    this.topCategory = this.categories.reduce((a, b) => b.prob > a.prob ? b : a);
  }
}
