import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ReactiveFormsModule, FormBuilder, FormGroup, Validators } from '@angular/forms';
import { Router } from '@angular/router';

@Component({
  selector: 'app-upload',
  standalone: true,
  imports: [CommonModule, ReactiveFormsModule],
  templateUrl: './upload.component.html',
  styleUrl: './upload.component.css'
})
export class UploadComponent {
  uploadForm: FormGroup;

  constructor(private fb: FormBuilder, private router: Router) {
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

  onSubmit() {
    if (this.uploadForm.valid) {
      console.log('Form data:', this.uploadForm.value);
      // Aqui iría la llamada al backend para procesar la imagen y los datos clínicos
      this.router.navigate(['/results']);
    } else {
      this.uploadForm.markAllAsTouched();
    }
  }
}
