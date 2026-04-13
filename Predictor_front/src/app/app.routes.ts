import { Routes } from '@angular/router';
import { HomeComponent } from './home/home.component';
import { UploadComponent } from './upload/upload.component';
import { ResultsComponent } from './results/results.component';
import { HowItWorksComponent } from './how-it-works/how-it-works.component';
import { FaqComponent } from './faq/faq.component';
import { LayoutComponent } from './layout/layout.component';

export const routes: Routes = [
  {
    path: '',
    component: LayoutComponent,
    children: [
      { path: '', component: HomeComponent },
      { path: 'upload', component: UploadComponent },
      { path: 'results', component: ResultsComponent },
      { path: 'how-it-works', component: HowItWorksComponent },
      { path: 'faq', component: FaqComponent },
    ]
  },
  { path: '**', redirectTo: '' }
];
