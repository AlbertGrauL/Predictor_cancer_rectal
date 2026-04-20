import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ReactiveFormsModule, FormBuilder, FormGroup, Validators } from '@angular/forms';
import { Router } from '@angular/router';
import { PredictionService } from '../prediction.service';

@Component({
  selector: 'app-upload',
  standalone: true,
  imports: [CommonModule, ReactiveFormsModule],
  templateUrl: './upload.component.html',
  styleUrl: './upload.component.css'
})
export class UploadComponent {
  uploadForm: FormGroup;
  selectedFile: File | null = null;
  isLoading = false;
  errorMessage = '';

  constructor(private fb: FormBuilder, private router: Router, private predictionService: PredictionService) {
    this.uploadForm = this.fb.group({
      sex: ['', Validators.required],
      tobacco: ['never', Validators.required],
      alcohol: ['none', Validators.required],
      radiotherapy: ['', Validators.required],
      diabetes: ['', Validators.required],
      formalCancer: ['', Validators.required],
      familyHistory: ['no', Validators.required],
      familyHistoryDetails: [''],
      bloodInStool: ['', Validators.required],
      rectorrhagia: ['', Validators.required],
      intestinalHabits: ['normal', Validators.required],
      tenesmus: ['', Validators.required]
    });
  }

  onFileSelected(event: Event) {
    const input = event.target as HTMLInputElement;
    if (input.files?.length) {
      this.selectedFile = input.files[0];
      this.errorMessage = '';
    }
  }

  onSubmit() {
    if (!this.uploadForm.valid) {
      this.uploadForm.markAllAsTouched();
      return;
    }
    if (!this.selectedFile) {
      this.errorMessage = 'Por favor seleccione una imagen.';
      return;
    }

    this.isLoading = true;
    this.errorMessage = '';

    const formData = new FormData();
    formData.append('image', this.selectedFile);

    const reader = new FileReader();
    reader.onload = () => {
      const imageDataUrl = reader.result as string;
      this.predictionService.predict(formData).subscribe({
        next: (result) => {
          this.isLoading = false;
          this.router.navigate(['/results'], {
            state: { result, imageDataUrl, clinicalData: this.uploadForm.value }
          });
        },
        error: () => {
          this.isLoading = false;
          this.errorMessage = 'Error al procesar la imagen. Verifique que el servidor esté activo en localhost:8000.';
        }
      });
    };
    reader.readAsDataURL(this.selectedFile);
  }
}
