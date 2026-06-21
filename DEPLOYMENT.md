# Déploiement — `aeg-snd.vercel.app`

Procédure pour mettre la plateforme en ligne **gratuitement** sur Vercel
(frontend) + Render (backend). Compte 15 minutes en tout.

---

## 1. Backend — Render.com (gratuit)

### 1.a — Crée un compte
1. Va sur **https://render.com**
2. Clique **"Get Started"** → **"Sign up with GitHub"**
3. Autorise Render à accéder à ton repo `adamelgbouri/testspyy`

### 1.b — Déploie le service
1. Dans le dashboard Render, clique **"New +"** → **"Blueprint"**
2. Sélectionne le repo `testspyy`
3. Render détecte automatiquement le fichier `web/backend/render.yaml`
4. Clique **"Apply"** — le build démarre (~3 min la première fois)

### 1.c — Récupère l'URL
Quand le statut passe à **"Live"** (point vert), copie l'URL :
```
https://aeg-snd-api.onrender.com
```
(le nom exact dépend de ce que Render choisit)

Teste-la dans le navigateur :
```
https://aeg-snd-api.onrender.com/api/health
```
Tu dois voir `{"status":"ok","version":"0.2.0"}`.

> ⚠️ Le free tier de Render endort le service après 15 min d'inactivité.
> Le premier appel après endormissement prend ~30s. Acceptable pour
> une démo perso.

---

## 2. Frontend — Vercel (gratuit)

### 2.a — Crée un compte
1. Va sur **https://vercel.com**
2. **"Continue with GitHub"** — autorise l'accès à `adamelgbouri/testspyy`

### 2.b — Importe le projet
1. Clique **"Add New..."** → **"Project"**
2. Sélectionne `testspyy` dans la liste
3. Dans l'écran **"Configure Project"** :
   - **Project Name** : `aeg-snd`  *(c'est ça qui donne ton URL `aeg-snd.vercel.app`)*
   - **Framework Preset** : `Next.js` (déjà détecté)
   - **Root Directory** : clique sur **"Edit"** → tape `web/frontend`
   - **Build Command** : laisse par défaut (lu depuis `vercel.json`)
   - **Install Command** : `npm install --legacy-peer-deps`
4. Déroule **"Environment Variables"** et ajoute :
   - Name : `NEXT_PUBLIC_API_URL`
   - Value : `https://aeg-snd-api.onrender.com`  *(l'URL Render de l'étape 1.c)*
5. Clique **"Deploy"**

### 2.c — Vérifie
Au bout de 1-2 minutes : **"Congratulations"** + lien clickable.

Ouvre `https://aeg-snd.vercel.app` — la landing page s'affiche.
Clique **"Open the dashboard"** — vérifie que les KPIs se chargent
(ils appellent ton backend Render).

---

## 3. Mises à jour automatiques

Désormais, à chaque `git push` sur la branche déployée :
- **Vercel** : redéploie le frontend en ~30s
- **Render** : redéploie le backend en ~3min

Tu peux voir les logs en temps réel dans les dashboards des deux services.

---

## 4. Limites des free tiers

| Service | Gratuit jusqu'à | Au-delà |
|---|---|---|
| **Vercel** | 100 GB bandwidth/mois | 20 \$/mois pour le plan Pro |
| **Render** Free | Sleep après 15 min, 750h/mois | 7 \$/mois pour `Starter` (always-on) |

Pour une démo perso + amis + recruteurs, tu restes très en dessous.

---

## 5. Petits "plus" optionnels

### Domaine custom (~10 €/an)
Une fois Vercel déployé, tu peux acheter `aeg-snd.com` chez Namecheap/OVH,
puis dans Vercel **"Settings → Domains"** : ajoute le domaine et configure
les DNS (Vercel guide pas-à-pas).

### Always-on backend (7 \$/mois)
Upgrade Render à **"Starter"** : plus jamais de cold-start.

### Vraies données temps réel
Le code utilise Yahoo (gratuit, retardé 15 min). Pour du vrai temps réel,
remplace par **Polygon.io** (~30 \$/mois) ou **TwelveData**.

---

## 6. Troubleshooting

| Symptôme | Cause | Fix |
|---|---|---|
| `API OFFLINE` dans le status bar | Render n'a pas démarré | Attends ~30s puis recharge |
| `CORS error` dans la console | URL backend mal configurée | Vérifie `NEXT_PUBLIC_API_URL` |
| Build Vercel échoue | Root dir mal réglé | `Settings → General → Root Directory` = `web/frontend` |
| Render dit "Build failed" | Python deps | Vérifie que `requirements.txt` est bien dans `web/backend/` |

Si tu bloques, prends une capture d'écran de l'erreur — on debug ensemble.
