# Design Spec — Lucy Chatbot Frontend (Angular)
**Date:** 2026-06-10  
**Project:** ai-agent-cnaps  
**Stack:** Angular 19, SCSS custom, Docker (Nginx prod / ng serve dev), port 3000 (prod) / 4200 (dev)

---

## Context

Le backend RAG CNaPS (FastAPI, port 8000) est fonctionnel mais n'a aucun frontend. Ce spec décrit l'interface chatbot "Lucy BOT" — un widget flottant Angular exposant le pipeline `/ask` aux usagers de la CNAPS.

---

## Stratégie Dev / Production

| Mode | Commande | Frontend | Backend |
|------|----------|----------|---------|
| Développement | `ng serve` local | localhost:4200 (hot-reload) | `docker compose up` → localhost:8000 |
| Production | `docker compose up` | localhost:3000 (Nginx) | localhost:8000 (FastAPI) |

**Principe clé :** le code Angular est identique dans les deux modes. `proxy.conf.json` redirige `/api/*` → `http://localhost:8000` en dev ; Nginx fait la même redirection en prod. Zéro changement de code au déploiement.

---

## Architecture

### Structure des dossiers
```
frontend/
├── src/
│   ├── app/
│   │   ├── components/
│   │   │   ├── chatbot-widget/         ← racine du widget (toggle global)
│   │   │   ├── floating-icon/          ← icône flottante bas-droite
│   │   │   ├── chat-header/            ← barre titre + boutons
│   │   │   ├── sidebar/                ← historique de session
│   │   │   ├── welcome-screen/         ← écran d'accueil + FAQ
│   │   │   ├── conversation-view/      ← zone de messages
│   │   │   ├── message-bubble/         ← bulle user / bot
│   │   │   ├── typing-indicator/       ← points clignotants
│   │   │   └── chat-input/             ← champ texte + bouton envoyer
│   │   ├── services/
│   │   │   ├── chat.service.ts         ← HTTP + state (Signals)
│   │   │   ├── session.service.ts      ← historique in-memory
│   │   │   └── typewriter.service.ts   ← effet typewriter mot par mot
│   │   └── models/
│   │       └── message.model.ts        ← interfaces TypeScript
│   ├── environments/
│   │   ├── environment.ts              ← { apiUrl: '/api' }
│   │   └── environment.prod.ts         ← { apiUrl: '/api' }
│   └── styles/
│       ├── _variables.scss             ← couleurs, tokens
│       └── _animations.scss            ← typewriter, typing dots
├── proxy.conf.json                     ← dev: /api → http://localhost:8000
├── Dockerfile                          ← multi-stage: node build → nginx
├── nginx.conf                          ← SPA routing + proxy /api → backend
└── .dockerignore
```

---

## Composants & Règles de gestion

### F1–F3 — FloatingIconComponent
- Position fixe `bottom: 24px; right: 24px` via SCSS
- Click toggle `isOpen` signal dans `ChatService`
- Si session existante (messages > 0) et page non rafraîchie → restaure la vue conversation

### F4–F8 — WelcomeScreenComponent
- Affiché si `messages.length === 0`
- Avatar Lucy (SVG), titre "Lucy — CNAPS"
- Message de bienvenue : "Bonjour, Comment puis-je vous aider ?"
- Boutons FAQ : liste définie dans `environment.ts` (configurable sans rebuild)
- Clic FAQ → soumet directement via `ChatService.sendMessage()`

### F9–F11 — SidebarComponent
- Panel latéral droit, toggle via icône dans le header
- Onglet "Récents" : 5 derniers échanges
- Onglet "Tous" : intégralité de la session
- Clic item → scroll vers le message correspondant

### F12–F13 — ChatInputComponent
- `<textarea>` auto-resize, désactivé si `isLoading`
- Bouton Envoyer désactivé si champ vide ou `isLoading`
- Entrée = soumission, Shift+Entrée = saut de ligne

### F14 — TypingIndicatorComponent
- 3 points avec animation CSS `bounce` séquentiel
- Visible uniquement quand `isLoading === true`

### F15 — ConversationViewComponent
- `MessageBubble` triés chronologiquement
- Auto-scroll vers le bas à chaque nouveau message
- Session conservée en Signals Angular (in-memory)
- Reset au refresh navigateur (pas de localStorage)

### F16 — TypewriterService
- Reçoit la réponse JSON complète de `/ask`
- Émet les mots un par un via `setInterval(30ms)` vers un signal `displayedText`
- L'indicateur de frappe (F14) disparaît quand l'effet commence

### F17 — Message "Information non disponible"
- Si HTTP error ou réponse vide → affiche : "Je n'ai pas trouvé d'information correspondante. Veuillez reformuler ou contacter directement la CNAPS."

### F18 — Bouton "Nouveau chat"
- `ChatService.resetSession()` vide le signal `messages`, retourne à WelcomeScreen

### F19 — Bouton "Fermer"
- Toggle `isOpen = false` — la session reste en mémoire pour restauration (F3)

---

## State Management (Angular Signals)

```typescript
// chat.service.ts
messages  = signal<Message[]>([]);
isOpen    = signal<boolean>(false);
isLoading = signal<boolean>(false);
activeView = computed(() =>
  this.messages().length === 0 ? 'welcome' : 'conversation'
);
```

```typescript
// message.model.ts
interface Message {
  id: string;
  role: 'user' | 'bot';
  content: string;
  timestamp: Date;
}
```

---

## API Integration

**Endpoint:** `POST /api/ask` (proxy → `http://localhost:8000/ask`)  
**Request:** `{ "message": "string" }`  
**Response:** `{ "answer": string, "metadata": {...}, "from_cache": boolean }`

```typescript
sendMessage(text: string): Observable<RAGResponse> {
  return this.http.post<RAGResponse>('/api/ask', { message: text });
}
```

### proxy.conf.json (dev)
```json
{
  "/api": {
    "target": "http://localhost:8000",
    "secure": false,
    "pathRewrite": { "^/api": "" }
  }
}
```

### nginx.conf (prod)
```nginx
location /api/ {
  proxy_pass http://backend:8000/;
}
location / {
  try_files $uri $uri/ /index.html;
}
```

---

## Docker

### Dockerfile (multi-stage)
```
Stage 1 — builder : node:22-alpine
  COPY package*.json && npm ci
  COPY src/ && ng build --configuration production

Stage 2 — runtime : nginx:alpine
  COPY --from=builder dist/frontend/browser /usr/share/nginx/html
  COPY nginx.conf /etc/nginx/conf.d/default.conf
  EXPOSE 80
```

### Ajout dans docker-compose.yml
```yaml
frontend:
  build: ./frontend
  ports:
    - "3000:80"
  depends_on:
    - app
  networks:
    - app-network
```

---

## Design Tokens (SCSS _variables.scss)
```scss
$color-primary   : #1a3a6e;   // header navy
$color-accent    : #1565c0;   // bouton envoyer
$color-bg        : #ffffff;
$color-bubble-user: #e3f2fd;
$color-bubble-bot : #f5f5f5;
$color-text      : #212121;
$widget-width    : 360px;
$widget-height   : 560px;
$border-radius   : 16px;
```

---

## Vérification end-to-end

### Dev (local)
```bash
cd frontend && ng serve --proxy-config proxy.conf.json
# Ouvrir http://localhost:4200
```

### Production (Docker)
```bash
docker compose up --build
# Ouvrir http://localhost:3000
```

### Scénarios à valider
1. Icône Lucy visible en bas à droite, clic ouvre la fenêtre
2. Écran d'accueil avec avatar et questions fréquentes
3. Clic FAQ → message soumis, typing indicator, réponse typewriter
4. Saisie manuelle + Entrée → même flux
5. Bouton "Nouveau chat" → retour écran d'accueil
6. Fermer la fenêtre → clic icône → session restaurée
7. Refresh navigateur → session réinitialisée
8. Panel historique → onglets Récents / Tous
