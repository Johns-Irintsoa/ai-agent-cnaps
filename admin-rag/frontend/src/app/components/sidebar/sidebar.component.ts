import { Component } from '@angular/core';

@Component({
  selector: 'app-sidebar',
  templateUrl: './sidebar.component.html',
  styleUrls: ['./sidebar.component.scss']
})
export class SidebarComponent {
  navItems = [
    { label: 'Tableau de bord', icon: 'dashboard',   route: '/dashboard' },
    { label: 'Documents',       icon: 'description', route: '/documents' },
    { label: 'Nouvel import',   icon: 'upload',      route: '/documents/upload' }
  ];
}
