import { Component } from '@angular/core';
import { CommonModule, DecimalPipe } from '@angular/common';
import { RouterLink } from '@angular/router';

interface SpecialistMetric {
  name: string;
  valLoss: number;
  auc: number;
  sensitivity: number;
  earlyStopEpoch: number;
  bestEpoch: number;
  trainSamples: number;
  valSamples: number;
  note?: string;
}

interface ComparisonRow {
  aspect: string;
  v1: string;
  v2: string;
  impact: 'positive' | 'neutral' | 'info';
}

interface BarDatum {
  label: string;
  value: number;
  color: string;
  pending?: boolean;
}

interface ChartBar {
  label: string;
  v1: number;
  v2?: number;
  v1Color: string;
}

@Component({
  selector: 'app-model-results',
  standalone: true,
  imports: [CommonModule, RouterLink, DecimalPipe],
  templateUrl: './model-results.component.html',
  styleUrl: './model-results.component.css'
})
export class ModelResultsComponent {
  v1Metrics: SpecialistMetric[] = [
    {
      name: 'Pólipos',
      valLoss: 0.0008,
      auc: 1.0000,
      sensitivity: 1.0000,
      bestEpoch: 26,
      earlyStopEpoch: 33,
      trainSamples: 4362,
      valSamples: 545
    },
    {
      name: 'Inflamación',
      valLoss: 0.0727,
      auc: 0.9930,
      sensitivity: 0.8537,
      bestEpoch: 9,
      earlyStopEpoch: 16,
      trainSamples: 4362,
      valSamples: 545
    },
    {
      name: 'Sangre',
      valLoss: 0.0753,
      auc: 0.9897,
      sensitivity: 0.9412,
      bestEpoch: 5,
      earlyStopEpoch: 12,
      trainSamples: 4362,
      valSamples: 545
    },
    {
      name: 'Negativos',
      valLoss: 0.4224,
      auc: 0.7177,
      sensitivity: 0.6029,
      bestEpoch: 6,
      earlyStopEpoch: 13,
      trainSamples: 5962,
      valSamples: 745,
      note: 'Rendimiento bajo: la clase agrupa categorías visualmente heterogéneas (ciego, píloro, márgenes de resección) que comparten rasgos con las clases positivas.'
    }
  ];

  comparisonRows: ComparisonRow[] = [
    { aspect: 'Nº de modelos',             v1: '4 clasificadores independientes',         v2: '1 modelo (arquitecturas rotadas)',             impact: 'positive' },
    { aspect: 'Estrategia',                v1: 'One-vs-Rest binario',                    v2: 'Multiclase unificada (3 clases)',              impact: 'positive' },
    { aspect: 'Clases',                    v1: 'pólipos / sangre / inflamación / negativos', v2: 'polipo / sano / otras_patologias',          impact: 'info' },
    { aspect: 'Arquitecturas probadas',    v1: 'EfficientNet-B0',                         v2: 'ResNet-50 · EfficientNet-B0 · DenseNet-121',  impact: 'positive' },
    { aspect: 'Optimizer',                 v1: 'Adam',                                    v2: 'AdamW (wd = 1×10⁻⁴)',                         impact: 'positive' },
    { aspect: 'Loss',                      v1: 'BCEWithLogitsLoss',                       v2: 'CrossEntropyLoss + pesos de clase',            impact: 'positive' },
    { aspect: 'Learning rate',             v1: 'Fijo',                                    v2: 'ReduceLROnPlateau (factor 0.5, patience 2)',   impact: 'positive' },
    { aspect: 'Fine-tuning',               v1: 'Backbone completo desde época 1',         v2: 'Freeze época 1 → descongelado completo',       impact: 'positive' },
    { aspect: 'Preprocesamiento',          v1: 'Sin máscara',                             v2: 'Bottom-left mask 30%×35% (elimina texto)',     impact: 'positive' },
    { aspect: 'Aumentación adicional',     v1: 'Básica',                                  v2: 'Random Erasing p=0.25 (regularización)',       impact: 'positive' },
    { aspect: 'Split Train/Val/Test',      v1: '80 / 10 / 10 %',                          v2: '70 / 15 / 15 %',                              impact: 'info' },
    { aspect: 'Early stopping patience',   v1: '7 épocas',                                v2: '5 épocas',                                    impact: 'neutral' },
    { aspect: 'Métricas objetivo',         v1: 'AUC-ROC, Sensibilidad',                   v2: 'F1-macro, Accuracy, AUC-ROC, PR-AUC',         impact: 'positive' },
    { aspect: 'Tracking de experimentos',  v1: 'MLflow (sqlite)',                          v2: 'JSON + manifests locales',                    impact: 'neutral' },
  ];

  getAucClass(auc: number): string {
    if (auc >= 0.99) return 'text-emerald-600 font-bold';
    if (auc >= 0.95) return 'text-blue-600 font-bold';
    if (auc >= 0.85) return 'text-amber-600 font-semibold';
    return 'text-red-600 font-semibold';
  }

  getAucBadge(auc: number): string {
    if (auc >= 0.99) return 'Excelente';
    if (auc >= 0.95) return 'Bueno';
    if (auc >= 0.85) return 'Aceptable';
    return 'Deficiente';
  }

  getAucBadgeClass(auc: number): string {
    if (auc >= 0.99) return 'bg-emerald-100 text-emerald-800';
    if (auc >= 0.95) return 'bg-blue-100 text-blue-800';
    if (auc >= 0.85) return 'bg-amber-100 text-amber-800';
    return 'bg-red-100 text-red-800';
  }

  getImpactClass(impact: string): string {
    if (impact === 'positive') return 'text-emerald-600';
    if (impact === 'neutral') return 'text-slate-500';
    return 'text-blue-600';
  }

  getImpactIcon(impact: string): string {
    if (impact === 'positive') return 'trending_up';
    if (impact === 'neutral') return 'remove';
    return 'info';
  }

  // ── SVG chart helpers ──────────────────────────────────────────────

  readonly chartW = 460;
  readonly chartH = 28;   // height per bar row
  readonly barMaxW = 300; // max bar pixel width (= value 1.0)
  readonly labelW = 110;
  readonly valW = 50;

  aucBars: BarDatum[] = [
    { label: 'Pólipos',      value: 1.0000, color: '#10b981' },
    { label: 'Inflamación',  value: 0.9930, color: '#3b82f6' },
    { label: 'Sangre',       value: 0.9897, color: '#6366f1' },
    { label: 'Negativos',    value: 0.7177, color: '#f59e0b' },
  ];

  sensBars: BarDatum[] = [
    { label: 'Pólipos',      value: 1.0000, color: '#10b981' },
    { label: 'Sangre',       value: 0.9412, color: '#6366f1' },
    { label: 'Inflamación',  value: 0.8537, color: '#3b82f6' },
    { label: 'Negativos',    value: 0.6029, color: '#f59e0b' },
  ];

  approachBars: ChartBar[] = [
    { label: 'Arquitecturas',    v1: 0.33,  v2: 1.0,  v1Color: '#3b82f6' },
    { label: 'Métricas eval.',   v1: 0.40,  v2: 1.0,  v1Color: '#3b82f6' },
    { label: 'Preprocesamiento', v1: 0.10,  v2: 0.85, v1Color: '#3b82f6' },
    { label: 'Regularización',   v1: 0.35,  v2: 0.80, v1Color: '#3b82f6' },
    { label: 'Balance clases',   v1: 0.20,  v2: 0.90, v1Color: '#3b82f6' },
    { label: 'Fine-tuning',      v1: 0.40,  v2: 0.85, v1Color: '#3b82f6' },
  ];

  barY(i: number): number { return i * (this.chartH + 8) + 4; }
  barPx(v: number): number { return Math.round(v * this.barMaxW); }
  svgH(n: number): number { return n * (this.chartH + 8) + 8; }
  textY(i: number): number { return this.barY(i) + this.chartH / 2 + 5; }

  // radar/spider chart (hexagon)
  readonly radarCx = 200;
  readonly radarCy = 200;
  readonly radarR  = 150;

  radarAxes = [
    { label: 'AUC-ROC',        v1: 0.93,  v2: 0.92 },
    { label: 'Sensibilidad',   v1: 0.84,  v2: 0.90 },
    { label: 'Robustez',       v1: 0.50,  v2: 0.85 },
    { label: 'Preprocesado',   v1: 0.15,  v2: 0.88 },
    { label: 'Arquitecturas',  v1: 0.33,  v2: 1.00 },
    { label: 'Métricas eval.', v1: 0.40,  v2: 1.00 },
  ];

  radarPoint(axisIndex: number, value: number): { x: number; y: number } {
    const n = this.radarAxes.length;
    const angle = (2 * Math.PI * axisIndex) / n - Math.PI / 2;
    return {
      x: this.radarCx + this.radarR * value * Math.cos(angle),
      y: this.radarCy + this.radarR * value * Math.sin(angle),
    };
  }

  radarGridPoint(axisIndex: number, r: number): { x: number; y: number } {
    const n = this.radarAxes.length;
    const angle = (2 * Math.PI * axisIndex) / n - Math.PI / 2;
    return { x: this.radarCx + r * Math.cos(angle), y: this.radarCy + r * Math.sin(angle) };
  }

  radarPolygon(version: 'v1' | 'v2'): string {
    return this.radarAxes
      .map((ax, i) => {
        const p = this.radarPoint(i, version === 'v1' ? ax.v1 : ax.v2);
        return `${p.x},${p.y}`;
      })
      .join(' ');
  }

  radarGridPolygon(scale: number): string {
    const r = this.radarR * scale;
    return this.radarAxes
      .map((_, i) => {
        const p = this.radarGridPoint(i, r);
        return `${p.x},${p.y}`;
      })
      .join(' ');
  }

  radarLabelPos(i: number): { x: number; y: number } {
    const n = this.radarAxes.length;
    const angle = (2 * Math.PI * i) / n - Math.PI / 2;
    const r = this.radarR + 24;
    return { x: this.radarCx + r * Math.cos(angle), y: this.radarCy + r * Math.sin(angle) };
  }
}
