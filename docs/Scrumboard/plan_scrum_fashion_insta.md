# Plan de Mise en Place Scrum - Fashion-Insta

## Vue d'Ensemble du Projet

Fashion-Insta est une plateforme de *Virtual Try-On* (VTON) pour l'industrie de la mode. Le projet se décompose en deux Programme Increments (PI) :

| PI       | Phase            | Durée                           | Objectif                             |
| -------- | ---------------- | ------------------------------- | ------------------------------------ |
| **PI 1** | PoC + Amorce MVP | 6 semaines (2 sprints x 3 sem)  | Validation technique + Features core |
| **PI 2** | MVP Final        | 12 semaines (4 sprints x 3 sem) | Produit livrable production          |

---

## 1. Contexte Métier & Besoins

### 1.1 Expression des Besoins Métiers

| Besoin | Catégorie | Priorité |
|--------|-----------|----------|
| **B1** - Compte utilisateur nominatif | Auth & Gestion | P0 |
| **B2** - Photo & collection garde-robe | VTON Garderobe | P0 |
| **B3** - Algorithme recommandation basé garde-robe | ML/AI | P0 |
| **B4** - Overlay virtuel sur photo utilisateur | VTON | P0 |
| **B5** - Changement couleur/style vêtement | Personnalisation | P1 |
| **B6** - Définition styles préférés | Préférences | P1 |
| **B7** - Référence marques préférées | Préférences | P1 |
| **B8** - Référence blogs/sites/influenceurs | Préférences | P2 |
| **B9** - Algorithme recommandation par préférences | ML/AI | P1 |
| **B10** - Avis sur les propositions | Feedback | P1 |
| **B11** - Désinscription RGPD | RGPD | P0 |
| **B12** - Accès/modification/suppression données | RGPD | P0 |
| **B13** - Durée de conservation | RGPD | P0 |
| **B14** - Purge automatique après inactivité | RGPD | P1 |

### 1.2 Projections Marketing

| Métrique | Valeur |
|----------|--------|
| Utilisateurs annuels | 400,000 |
| Utilisateurs actifs | ~20,000-50,000 |
| Impact CA Web (24 mois) | +14% |
| Impact CA Magasin (24 mois) | +4% |
| Target latence VTON | 3-5 secondes |

---

## 2. Configuration Scrum

### 2.1 Structure temporelle

| Unité      | Durée                      |
| ---------- | -------------------------- |
| **Sprint** | 3 semaines                 |
| **PI**     | 4 sprints (12 semaines)    |
| **PoC**    | Max 2 sprints (6 semaines) |
| **MVP**    | 6 sprints (PI 1 + PI 2)    |

### 2.2 Équipe Scrum

| Rôle              | Profil             | Coût Journalier | Engagement                |
| ----------------- | ------------------ | --------------- | ------------------------- |
| **Product Owner** | Interne dédié      | -               | 100%                      |
| **Scrum Master**  | Interne ou externe | -               | 50% (full PI 1, 30% PI 2) |
| **Developers**    | 4 profils Data     | 1,480 €/jour    | 100%                      |

**Composition équipe :**
- 1x Data Scientist (350 €/j)
- 1x Data Engineer (370 €/j)
- 1x ML Ops Engineer (360 €/j)
- 1x Tech Lead Data (400 €/j)

### 2.3 Événements Scrum

| Événement | Durée | Fréquence | Jour |
|-----------|-------|-----------|------|
| **Sprint Planning** | 4h | Sprint (S1) | Semaine 1, lundi |
| **Daily Scrum** | 15 min | Quotidien | Tous les jours |
| **Weekly Sync** | 1h | Hebdomadaire | Mercredi |
| **Sprint Review** | 2h | Sprint (S3) | Semaine 3, vendredi |
| **Sprint Retrospective** | 1.5h | Sprint (S3) | Semaine 3, vendredi |
| **PI Planning** | 1 jour | PI (Q3) | Début chaque PI |
| **Scrum of Scrums** | 30 min | Hebdomadaire | Mardi (avec experts métier) |

---

## 3. Product Backlog - Besoins Métiers Cartographiés

### 3.1 Epics & User Stories

#### Epic E1: Gestion Utilisateur & RGPD (B1, B11-B14)

| US | Description | Estimation |
|----|------------|------------|
| US 1.1 | Inscription/authentification compte nominatif | 5 SP |
| US 1.2 | Connexion JWT + refresh tokens | 3 SP |
| US 1.3 | Dashboard gestion données personnelles | 5 SP |
| US 1.4 | Demande suppression compte (RGPD) | 3 SP |
| US 1.5 | Export données utilisateur (portabilité) | 5 SP |
| US 1.6 | Configuration durée conservation | 3 SP |
| US 1.7 | Job purge automatique (inactivité) | 8 SP |

#### Epic E2: Garderobe Utilisateur (B2)

| US | Description | Estimation |
|----|------------|------------|
| US 2.1 | Upload photo vetement via mobile | 3 SP |
| US 2.2 | Catalogue personnel (CRUD) | 5 SP |
| US 2.3 | Tagging automatique catégories | 8 SP |
| US 2.4 | Suppression multiple photos | 2 SP |

#### Epic E3: Virtual Try-On - Garde-robe (B3-B4)

| US | Description | Estimation |
|----|------------|------------|
| US 3.1 | Pipeline VTON complet (garment → output) | 13 SP |
| US 3.2 | API recommandation basée garderobe | 8 SP |
| US 3.3 | Overlay vetement sur photo utilisateur | 8 SP |
| US 3.4 | Changement couleur vetement | 8 SP |
| US 3.5 | Changement style (manches courtes/longues) | 13 SP |

#### Epic E4: Système de Préférences (B6-B9)

| US | Description | Estimation |
|----|------------|------------|
| US 4.1 | Interface selection styles prefers | 3 SP |
| US 4.2 | Interface selection marques preferees | 3 SP |
| US 4.3 | Interface ajout blogs/sites/influenceurs | 5 SP |
| US 4.4 | Algorithme recommandation hybride | 13 SP |

#### Epic E5: Feedback & Notes (B10)

| US | Description | Estimation |
|----|------------|------------|
| US 5.1 | Systeme notation propositions | 3 SP |
| US 5.2 | Collecte feedback qualitatifs | 3 SP |
| US 5.3 | Dashboard analytics feedback | 5 SP |

---

## 4. Plan de Release - 2 PIs

### 4.1 PI 1 - Sprint 1 à 4 (Semaines 1-12)

```
PI 1 Timeline:
[Sprint 1: S1-S3] → [Sprint 2: S4-S6] → [Sprint 3: S7-S9] → [Sprint 4: S10-S12]
     PoC                   PoC              MVP Start            MVP v0.5
```

| Sprint | Focus | Objectif | Livrables |
|--------|-------|----------|-----------|
| **S1** | Setup & PoC | Infrastructure & MVP pipeline VTON | Repo code, pipeline base, env Azure |
| **S2** | PoC Final | Validation technique | Demo PoC, métriques qualité, rapport technique |
| **S3** | MVP Core | API & Auth | Auth JWT, API REST utilisateurs, CRUD garderobe |
| **S4** | MVP v0.5 | VTON basic | Pipeline VTON integre, endpoint inference, premier test utilisateur |

**Definition of Done PoC (Sprints 1-2):**
- [ ] Pipeline VTON execitable de bout en bout
- [ ] Latence inference < 10s (target < 5s)
- [ ] Métriques qualité collectées (SSIM, FID)
- [ ] Documentation technique
- [ ] Demo validée par stakeholders

### 4.2 PI 2 - Sprint 5 à 8 (Semaines 13-24)

```
PI 2 Timeline:
[Sprint 5: S13-S15] → [Sprint 6: S16-S18] → [Sprint 7: S19-S21] → [Sprint 8: S22-S24]
    MVP v1               MVP v2              Pre-prod             Production
```

| Sprint | Focus | Objectif | Livrables |
|--------|-------|----------|-----------|
| **S5** | VTON Avancé | Personnalisation | Changement couleur/style, recommendations hybrides |
| **S6** | Feedback & Prefs | Système preferences | Interface prefs, algo recommandations, feedback |
| **S7** | RGPD & Perf | Conformité & optimisation | Purge auto, optimisations performance, tests charge |
| **S8** | Production | Release MVP | Déploiement Azure L4, monitoring, documentation finale |

**Definition of Done MVP (Sprint 8):**
- [ ] Tests d'intégration passés
- [ ] Tests de performance (500+ req/min)
- [ ] Monitoring & alerting opérationnel
- [ ] Documentation utilisateur complète
- [ ] Conformité RGPD validée
- [ ] Uptime target 99%

---

## 5. Matrice RACI - Responsabilités par Profil

### 5.1 Répartition par Phase

| Activité | DS | DE | MLOps | TL | PO | SM |
|----------|:--:|:--:|:-----:|:--:|:--:|:--:|
| **Architecture & Design** | C | C | C | A | I | I |
| **Développement Pipeline ML** | A | R | R | C | I | I |
| **Déploiement Modeles** | I | C | A/R | C | I | I |
| **API & Backend** | I | A | R | C | I | I |
| **Infrastructure Cloud** | I | C | A | R | I | I |
| **Tests & Validation** | R | R | R | A | C | I |
| **Documentation** | R | R | R | A | C | I |
| **Backlog Management** | I | I | I | C | A | R |

*A = Accountable, R = Responsible, C = Consulted, I = Informed*

### 5.2 Taux de Staffing par Phase

| Profil | PI 1 (S1-S4) | PI 2 (S5-S8) | Post-MVP |
|--------|--------------|--------------|----------|
| Data Scientist | 100% | 80% | 50% |
| Data Engineer | 100% | 80% | 30% |
| ML Ops Engineer | 100% | 100% | 80% |
| Tech Lead Data | 50% | 30% | 20% |
| Scrum Master | 50% | 30% | 20% |

**Justification:**
- **DS/DE/MLOps 100% en PI 1** : Phase critique PoC nécessitant expertise technique intensive
- **MLOps 100% en PI 2** : Déploiement production et monitoring requièrent présence continue
- **TL décroissant** : Rôle de validation/cadrage, moins hands-on en delivery
- **DS/DE passent à 80% post-MVP** : Maintenance vs développement, focus ops

---

## 6. Budget Estimation

### 6.1 Coût Équipe (8 sprints x 3 sem = 24 semaines)

| Profil | Jours (24 sem) | Coût/Unitaire | Total |
|--------|----------------|---------------|-------|
| Data Scientist | 120 | 350 € | 42,000 € |
| Data Engineer | 120 | 370 € | 44,400 € |
| ML Ops Engineer | 120 | 360 € | 43,200 € |
| Tech Lead Data | 80 | 400 € | 32,000 € |
| **Total Équipe** | | | **161,600 €** |

### 6.2 Coût Infrastructure GPU

| Periode | GPU | Configuration | Coût/Mois | Total |
|---------|-----|---------------|-----------|-------|
| **PI 1** (3 mois) | L4 | 1 instance | ~300 € | ~900 € |
| **PI 2** (3 mois) | L4 | 1 instance | ~300 € | ~900 € |
| **Production Y1** | L4 | 1 instance | ~300 € | ~3,600 € |
| **Peak Season** | L4 x2 | Nov-Dec | +300 € | +600 € |

**Total Infrastructure (12 mois) :** ~6,000 €

### 6.3 Coûts Azure Services Associés

| Service | Utilisation | Coût/Mois | Total (12 mois) |
|---------|-------------|-----------|------------------|
| Azure Blob Storage | Images utilisateurs | ~50 € | 600 € |
| Azure Cosmos DB | Métadonnées | ~80 € | 960 € |
| Azure App Service | API/FE | ~100 € | 1,200 € |
| Azure Key Vault | Secrets | ~20 € | 240 € |
| Azure Monitor | Monitoring | ~50 € | 600 € |
| **Sous-total** | | | **3,600 €** |

### 6.4 Budget Total Projet

| Composant | Coût |
|-----------|------|
| Équipe (8 sprints) | 161,600 € |
| Infrastructure GPU | 6,000 € |
| Services Azure | 3,600 € |
| **TOTAL** | **171,200 €** |

---

## 7. Matrice Traçabilité Besoins → Sprints

| Besoin Métier | Sprint(s) | Epic | Critère Acceptation |
|---------------|-----------|------|---------------------|
| B1 - Compte nominatif | S3 | E1 | Auth fonctionnelle, token JWT |
| B2 - Photo & collection | S3-S4 | E2 | CRUD garderobe, upload image |
| B3 - Algo recommandation garderobe | S1-S4 | E3 | Pipeline complet, latence < 5s |
| B4 - Overlay virtuel | S4 | E3 | Image resultat visible |
| B5 - Changement couleur/style | S5 | E3 | Parametrage color/style fonctionnel |
| B6 - Styles prefers | S6 | E4 | Selection multi-styles sauvegardee |
| B7 - Marques preferees | S6 | E4 | Selection marques fonctionnelle |
| B8 - Blogs/influenceurs | S6 | E4 | Liste editable sauvegardee |
| B9 - Algo recommandation prefs | S6 | E4 | Hybrid recommendation functional |
| B10 - Avis/feedback | S6 | E5 | Systeme notation operatif |
| B11 - Desinscription | S7 | E1 | Account deletion pipeline |
| B12 - Acces/suppression donnees | S7 | E1 | Dashboard RGPD complet |
| B13 - Duree conservation | S7 | E1 | Parametrage configurable |
| B14 - Purge automatique | S7 | E1 | Job cron inactivite 12 mois |

---

## 8. Métriques & KPIs

### 8.1 KPIs Projet

| KPI | Cible |
|-----|-------|
| **Sprint Goal Completion** | > 80% |
| **Velocity** | 20-30 story points/sprint |
| **Lead Time** | < 5 jours |
| **Cycle Time** | < 3 jours |

### 8.2 KPIs Techniques

| Métrique | PoC (S1-S2) | MVP (S8) |
|----------|-------------|-----------|
| Latence inference VTON | < 10s | < 5s |
| Uptime | 95% | 99% |
| Temps de déploiement | - | < 30 min |
| Couverture tests | > 60% | > 80% |

### 8.3 KPIs Métier

| Métrique | Cible |
|----------|-------|
| Utilisateurs annuels | 400,000 |
| Conversion VTON → Achat | > 10% |
| Satisfaction utilisateur | > 4/5 |

---

## 9. Risques & Mitigation

| Risque | Impact | Probabilité | Mitigation |
|--------|--------|-------------|------------|
| **Retard PoC (modèle ML)** | Élevé | Haute | Buffer 1 sprint, scope ajustement |
| **Coûts GPU exceed budget** | Moyen | Moyenne | Reserved instances, auto-scaling L4 |
| **Disponibilité profils Data** | Moyen | Haute | Recrutement 2 mois avant |
| **Non-conformité RGPD** | Élevé | Moyenne | Audit juridique sprint 7 |
| **Scalabilité 400k users** | Élevé | Moyenne | Design cloud-natif, load testing S7 |

---

## 10. Roadmap Visuelle

```
PI 1 (12 sem)                          PI 2 (12 sem)
[S1] [S2] [S3] [S4]                    [S5] [S6] [S7] [S8]
 |     |    |    |                       |    |    |    |
 PoC  PoC  MVP   MVP v0.5              v1   v2   Pre-Prod Prod
 |____|____|____|                       |____|____|____|
      Release PoC                             Release MVP
```

---

## 11. Points de Vigilance

### 11.1 Contraintes Techniques

- **Azure L4** : Recommandé pour inference (~9-12GB VRAM requis)
- **Scalabilité** : Cible 500+ req/min en production
- **Coûts** : ~6,000 €/an GPU + ~3,600 €/an services Azure

### 11.2 Points d'attention Métier

- Intégrer les gains marketing (14% CA Web, 4% CA magasin)
- 400k utilisateurs annuels nécessitent design cloud-natif
- Conformité RGPD complète (B11-B14)
- Feedback loop pour améliorer les recommandations

---

## 12. Prochaines Étapes

1. **Semaine -2 à 0 (Pre-Sprint)** :
   - Finaliser Product Backlog priorisé
   - configurer environnement dev Azure
   - Sessions story mapping avec stakeholders

2. **Sprint 1** :
   - Kick-off PI Planning
   - Setup infrastructure, premier pipeline VTON

3. **Revue PI** (fin PI 1) :
   - Démonstration PoC
   - Validation stakeholders
   - Ajustement backlog PI 2