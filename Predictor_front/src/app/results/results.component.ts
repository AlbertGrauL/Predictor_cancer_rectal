import { Component } from '@angular/core';
import { CommonModule, PercentPipe } from '@angular/common';
import { RouterLink, Router } from '@angular/router';
import { PredictionResult } from '../prediction.service';

interface CategoryEntry {
  key: string;
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

  private labelMap: Record<string, string> = {
    polipos: 'Pólipos',
    sangre: 'Sangre',
    inflamacion: 'Inflamación',
    negativos: 'Negativo',
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

    const scores = result.image_scores ?? {
      polipos: result.polipos,
      sangre: result.sangre,
      inflamacion: result.inflamacion,
      negativos: result.negativos,
    };

    this.categories = (Object.keys(scores) as string[]).map(k => ({
      key: k,
      label: this.labelMap[k] ?? k,
      prob: (scores as any)[k] as number,
    }));
    this.topCategory = this.categories.reduce((a, b) => b.prob > a.prob ? b : a);
  }

  get clinicalLevel(): string {
    const level = this.result?.clinical_risk?.level ?? 'bajo';
    const map: Record<string, string> = {
      bajo: 'Bajo', moderado: 'Moderado', alto: 'Alto', muy_alto: 'Muy alto'
    };
    return map[level] ?? level;
  }

  get clinicalLevelClass(): string {
    const level = this.result?.clinical_risk?.level ?? 'bajo';
    return {
      bajo: 'text-emerald-700 bg-emerald-50 border-emerald-200',
      moderado: 'text-amber-700 bg-amber-50 border-amber-200',
      alto: 'text-orange-700 bg-orange-50 border-orange-200',
      muy_alto: 'text-red-700 bg-red-50 border-red-200',
    }[level] ?? '';
  }

  get fusionLevel(): string {
    const level = this.result?.fusion?.level ?? 'bajo';
    const map: Record<string, string> = {
      bajo: 'Bajo', moderado: 'Moderado', alto: 'Alto', muy_alto: 'Muy alto'
    };
    return map[level] ?? level;
  }

  get fusionLevelClass(): string {
    const level = this.result?.fusion?.level ?? 'bajo';
    return {
      bajo: 'text-emerald-700 bg-emerald-50 border-emerald-200',
      moderado: 'text-amber-700 bg-amber-50 border-amber-200',
      alto: 'text-orange-700 bg-orange-50 border-orange-200',
      muy_alto: 'text-red-700 bg-red-50 border-red-200',
    }[level] ?? '';
  }

  get fusionBarWidth(): string {
    return ((this.result?.fusion?.score ?? 0) * 100) + '%';
  }

  get clinicalBarWidth(): string {
    return ((this.result?.clinical_risk?.score ?? 0) * 100) + '%';
  }
}
