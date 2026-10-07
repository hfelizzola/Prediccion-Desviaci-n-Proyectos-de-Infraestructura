#!/usr/bin/env bash
# =============================================================================
# versionado.sh — Gestión de versiones del proyecto (código en git + artefactos)
#
# El código, las consultas SQL, los notebooks y los resultados pequeños se
# versionan en git. Los datos y modelos grandes quedan fuera de git; cada
# versión registra su huella SHA-256 en MANIFEST/artifacts.sha256 y puede
# empaquetarse para GitHub Releases o Zenodo.
#
# Uso (Git Bash en Windows, Linux, macOS o Colab con `!bash ...`):
#   bash scripts/versionado.sh <comando> [argumentos]
#   bash scripts/versionado.sh ayuda
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MAIN_BRANCH="main"
MANIFEST_DIR="MANIFEST"
MANIFEST_FILE="$MANIFEST_DIR/artifacts.sha256"
MANIFEST_INFO="$MANIFEST_DIR/artifacts.info"
DIST_DIR="${VERSIONADO_DIST:-dist}"   # p. ej. VERSIONADO_DIST=~/Downloads para no ocupar espacio en Drive
# Carpetas cuyos archivos ignorados se consideran artefactos de la versión
ARTIFACT_PATHS=("Data" "Models" "Models V2" "terridata" "Parameters" "outputs")
MAX_FILE_MB=50     # bloquea archivos versionados más grandes que esto
WARN_FILE_MB=10    # avisa por encima de este tamaño
SEMVER_RE='^v[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.-]+)?$'
COMMIT_RE='^(feat|fix|docs|refactor|perf|test|data|exp|build|ci|chore|style|release|merge)(\([^)]+\))?!?: .+'

# Patrones de secretos (ERE). El "\\?" admite comillas escapadas dentro de .ipynb.
SECRET_PATTERNS=(
  'password[[:space:]]*=[[:space:]]*\\?["'"'"'][^"'"'"'$]{4,}\\?["'"'"']'
  '(api_?key|app_?token|secret|token|API_?KEY|APP_?TOKEN|SECRET|TOKEN)[[:space:]]*=[[:space:]]*\\?["'"'"'][A-Za-z0-9]*[a-z][A-Za-z0-9]*[0-9][A-Za-z0-9]*\\?["'"'"']'
  'AKIA[0-9A-Z]{16}'
  'gh[pousr]_[A-Za-z0-9]{36}'
  '-----BEGIN [A-Z ]*PRIVATE KEY-----'
)

# --- utilidades --------------------------------------------------------------
log()  { printf '\033[1;34m[versionado]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[aviso]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

sha256() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$@"; else shasum -a 256 "$@"; fi; }

require_repo() { [[ -e .git ]] || die "No hay repositorio git en $ROOT. Ejecuta: bash scripts/versionado.sh init"; }

current_branch() { git symbolic-ref --short HEAD 2>/dev/null || echo "(detached)"; }

require_clean() {
  [[ -z "$(git status --porcelain)" ]] || die "Hay cambios sin guardar. Usa 'save' antes de continuar."
}

# Archivos ignorados por git dentro de las carpetas de artefactos + *.joblib de la raíz
list_artifacts() {
  git ls-files --others --ignored --exclude-standard -z -- "${ARTIFACT_PATHS[@]}" ':(glob)*.joblib' \
    | tr '\0' '\n' \
    | grep -v -E '(^|/)(\.DS_Store|desktop\.ini|Thumbs\.db)$|\.ipynb_checkpoints/' \
    | LC_ALL=C sort || true
}

install_hook() {
  local hook
  hook="$(git rev-parse --git-path hooks)/pre-commit"
  cat > "$hook" <<'EOF'
#!/usr/bin/env bash
# Instalado por scripts/versionado.sh: bloquea commits con secretos o archivos grandes
# y, si pre-commit está disponible, ejecuta ruff / nbstripout (.pre-commit-config.yaml).
root="$(git rev-parse --show-toplevel)"
bash "$root/scripts/versionado.sh" check --staged || exit 1
if [ -f "$root/.pre-commit-config.yaml" ] && command -v pre-commit >/dev/null 2>&1; then
  exec pre-commit run
fi
EOF
  chmod +x "$hook"
  log "Hook pre-commit instalado ($hook)"
}

update_changelog() {
  local version="$1" today
  today="$(date +%Y-%m-%d)"
  [[ -f CHANGELOG.md ]] || printf '# Changelog\n\n## [Unreleased]\n' > CHANGELOG.md
  if grep -q "^## \[$version\]" CHANGELOG.md; then return 0; fi
  awk -v v="$version" -v d="$today" '
    /^## \[Unreleased\]/ && !done { print; print ""; print "## [" v "] - " d; done = 1; next }
    { print }' CHANGELOG.md > CHANGELOG.md.tmp
  mv CHANGELOG.md.tmp CHANGELOG.md
  log "CHANGELOG.md: contenido de [Unreleased] asignado a $version"
}

# --- comandos ----------------------------------------------------------------
cmd_init() {
  local url="${1:-}"
  if [[ -e .git ]]; then
    log "El repositorio ya existe en $ROOT"
  else
    git init -b "$MAIN_BRANCH"
  fi
  git config core.longpaths true
  # LF en la copia de trabajo: evita reescrituras CRLF en Drive y diffs falsos frente a Colab/Linux
  git config core.autocrlf false
  git config core.eol lf
  git config user.name >/dev/null || die "Configura tu nombre: git config --global user.name \"Tu Nombre\""
  git config user.email >/dev/null || die "Configura tu correo: git config --global user.email tu@correo"
  install_hook
  [[ -z "$url" ]] || cmd_remote "$url"
  log "Listo. Siguiente paso: bash scripts/versionado.sh snapshot v1.0.0 \"Versión de la tesis doctoral\""
}

cmd_remote() {
  local url="${1:-}"
  [[ -n "$url" ]] || die "Uso: remote <url-del-repo-en-github>"
  require_repo
  if git remote get-url origin >/dev/null 2>&1; then
    git remote set-url origin "$url"
  else
    git remote add origin "$url"
  fi
  log "Remoto origin -> $url"
}

# check [--staged]: busca secretos y archivos grandes en lo que entraría a git
cmd_check() {
  require_repo
  local files fail=0 f size_mb
  if [[ "${1:-}" == "--staged" ]]; then
    files="$(git diff --cached --name-only --diff-filter=ACMR)"
  else
    files="$(git ls-files --cached --others --exclude-standard)"
  fi
  [[ -n "$files" ]] || { log "check: no hay archivos que revisar"; return 0; }

  local grep_args=()
  for p in "${SECRET_PATTERNS[@]}"; do grep_args+=(-e "$p"); done

  while IFS= read -r f; do
    [[ -f "$f" && "$f" != "scripts/versionado.sh" ]] || continue
    if grep -nIE "${grep_args[@]}" "$f" 2>/dev/null | cut -c1-160 | sed "s|^|  $f:|" | grep .; then
      warn "Posible secreto en $f (usa variables de entorno / .env)"
      fail=1
    fi
    size_mb=$(( $(wc -c < "$f") / 1048576 ))
    if (( size_mb >= MAX_FILE_MB )); then
      warn "$f pesa ${size_mb} MB (máximo ${MAX_FILE_MB} MB): agrégalo a .gitignore y al manifiesto"
      fail=1
    elif (( size_mb >= WARN_FILE_MB )); then
      warn "$f pesa ${size_mb} MB; considera dejarlo fuera de git"
    fi
  done <<< "$files"

  if [[ -e .env ]] && git ls-files --error-unmatch .env >/dev/null 2>&1; then
    warn ".env está versionado: ejecuta 'git rm --cached .env'"
    fail=1
  fi
  (( fail == 0 )) || die "check falló: corrige los problemas anteriores antes de confirmar"
  log "check OK: sin secretos ni archivos excesivos"
}

cmd_manifest() {
  require_repo
  local version="${1:-sin-version}" list n total=0 f
  list="$(list_artifacts)"
  if [[ -z "$list" ]]; then warn "No hay artefactos fuera de git"; return 0; fi
  n="$(printf '%s\n' "$list" | wc -l | tr -d ' ')"
  mkdir -p "$MANIFEST_DIR"
  log "Calculando SHA-256 de $n artefactos (en Google Drive puede tardar varios minutos)..."
  : > "$MANIFEST_FILE.tmp"
  while IFS= read -r f; do
    sha256 "$f" >> "$MANIFEST_FILE.tmp"
    total=$(( total + $(wc -c < "$f") ))
  done <<< "$list"
  mv "$MANIFEST_FILE.tmp" "$MANIFEST_FILE"
  {
    echo "version=$version"
    echo "fecha=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "commit_base=$(git rev-parse --short HEAD 2>/dev/null || echo 'ninguno')"
    echo "archivos=$n"
    echo "bytes_totales=$total"
  } > "$MANIFEST_INFO"
  log "Manifiesto escrito en $MANIFEST_FILE ($n archivos, $(( total / 1048576 )) MB)"
}

cmd_verify() {
  [[ -f "$MANIFEST_FILE" ]] || die "No existe $MANIFEST_FILE"
  log "Verificando artefactos locales contra $MANIFEST_FILE ..."
  if sha256 -c --quiet "$MANIFEST_FILE"; then
    log "Todos los artefactos coinciden con la versión registrada"
  else
    die "Hay artefactos modificados o ausentes respecto al manifiesto"
  fi
}

cmd_save() {
  local msg="${1:-}"
  [[ -n "$msg" ]] || die "Uso: save \"tipo(alcance): descripción\"  (ej. \"refactor(etl): extraer cliente Socrata\")"
  require_repo
  [[ "$msg" =~ $COMMIT_RE ]] || warn "El mensaje no sigue Conventional Commits (feat|fix|docs|refactor|data|exp|...: descripción)"
  cmd_check
  git add -A
  if git diff --cached --quiet; then log "No hay cambios que guardar"; return 0; fi
  git commit -m "$msg"
}

cmd_snapshot() {
  local version="${1:-}" msg="${2:-}"
  [[ "$version" =~ $SEMVER_RE ]] || die "Versión inválida '$version'. Usa vMAYOR.MENOR.PARCHE (ej. v1.0.0, v2.0.0-rc.1)"
  [[ -n "$msg" ]] || die "Uso: snapshot $version \"descripción de la versión\""
  require_repo
  if git rev-parse -q --verify "refs/tags/$version" >/dev/null; then die "El tag $version ya existe"; fi
  [[ "$(current_branch)" == "$MAIN_BRANCH" ]] || warn "Estás en '$(current_branch)'; las versiones suelen etiquetarse en $MAIN_BRANCH"

  cmd_check
  cmd_manifest "$version"
  update_changelog "$version"
  git add -A
  if ! git diff --cached --quiet; then git commit -m "release: $version - $msg"; fi
  git tag -a "$version" -m "$msg"
  log "Versión $version creada en el commit $(git rev-parse --short HEAD)"
  log "Para publicar: bash scripts/versionado.sh publish   |   artefactos: bash scripts/versionado.sh archive $version"
}

cmd_archive() {
  require_repo
  local version="${1:-}" out list
  [[ -n "$version" ]] || version="$(git describe --tags --abbrev=0 2>/dev/null)" || die "Indica la versión: archive vX.Y.Z"
  [[ -f "$MANIFEST_FILE" ]] || die "Primero genera el manifiesto (snapshot o manifest)"
  mkdir -p "$DIST_DIR"
  out="$DIST_DIR/${version}-artifacts.tar.gz"
  list="$DIST_DIR/.filelist"
  list_artifacts > "$list"
  printf '%s\n' "$MANIFEST_FILE" "$MANIFEST_INFO" >> "$list"
  log "Empaquetando $(wc -l < "$list" | tr -d ' ') archivos en $out ..."
  tar -czf "$out" -T "$list"
  rm -f "$list"
  ( cd "$DIST_DIR" && sha256 "$(basename "$out")" > "$(basename "$out").sha256" )
  log "Listo: $out ($(( $(wc -c < "$out") / 1048576 )) MB). Súbelo como asset del release $version en GitHub o a Zenodo."
}

cmd_branch() {
  local name="${1:-}" base="${2:-$MAIN_BRANCH}"
  [[ -n "$name" ]] || die "Uso: branch <nombre> [rama-base]   (crea feature/<nombre>)"
  require_repo
  [[ "$name" == */* ]] || name="feature/$name"
  git switch -c "$name" "$base"
  log "Trabajando en $name. Al terminar: bash scripts/versionado.sh finish"
}

cmd_finish() {
  require_repo
  local branch
  branch="$(current_branch)"
  [[ "$branch" != "$MAIN_BRANCH" ]] || die "Ya estás en $MAIN_BRANCH; 'finish' se usa desde una rama de trabajo"
  require_clean
  git switch "$MAIN_BRANCH"
  git merge --no-ff "$branch" -m "merge: $branch"
  log "$branch integrada en $MAIN_BRANCH. Para borrarla: git branch -d $branch"
}

cmd_publish() {
  require_repo
  git remote get-url origin >/dev/null 2>&1 || die "No hay remoto. Usa: remote https://github.com/<usuario>/<repo>.git"
  local branch
  branch="$(current_branch)"
  git push -u origin "$branch"
  git push origin --tags
  log "Publicado $branch y tags en $(git remote get-url origin)"
}

cmd_status() {
  require_repo
  echo "Rama actual : $(current_branch)"
  echo "Última vers.: $(git describe --tags --abbrev=0 2>/dev/null || echo 'ninguna')"
  echo "Remoto      : $(git remote get-url origin 2>/dev/null || echo 'no configurado')"
  [[ -f "$MANIFEST_INFO" ]] && echo "Manifiesto  : $(tr '\n' ' ' < "$MANIFEST_INFO")"
  echo "Versiones   :"
  git tag -n1 --sort=-creatordate | head -10 | sed 's/^/  /'
  echo "Cambios pendientes:"
  git status --short | head -25 | sed 's/^/  /'
}

cmd_help() {
  cat <<'EOF'
Uso: bash scripts/versionado.sh <comando> [argumentos]

Configuración
  init [url]                  Inicializa el repo (rama main), instala hook pre-commit y remoto opcional
  remote <url>                Configura/actualiza el remoto origin (GitHub)

Trabajo diario
  status                      Rama, última versión, remoto y cambios pendientes
  check                       Busca secretos y archivos grandes en lo que entraría a git
  save "<mensaje>"            check + commit de todos los cambios (Conventional Commits)
  branch <nombre> [base]      Crea la rama feature/<nombre> desde main
  finish                      Integra la rama actual en main (merge --no-ff)

Versiones
  snapshot <vX.Y.Z> "<msg>"   check + manifiesto SHA-256 + CHANGELOG + commit + tag anotado
  manifest [vX.Y.Z]           Regenera MANIFEST/artifacts.sha256 de datos/modelos fuera de git
  verify                      Comprueba que los artefactos locales coinciden con el manifiesto
  archive [vX.Y.Z]            Empaqueta los artefactos en dist/<versión>-artifacts.tar.gz
  publish                     Sube la rama actual y todos los tags a origin

Convención de versiones para este proyecto de investigación
  MAYOR  cambia los resultados reportados (datos, muestra, metodología)
  MENOR  agrega análisis o funcionalidades sin cambiar resultados previos
  PARCHE correcciones que no alteran resultados
EOF
}

main() {
  local cmd="${1:-ayuda}"
  shift || true
  case "$cmd" in
    init)      cmd_init "$@" ;;
    remote)    cmd_remote "$@" ;;
    check)     cmd_check "$@" ;;
    save)      cmd_save "$@" ;;
    snapshot)  cmd_snapshot "$@" ;;
    manifest)  cmd_manifest "$@" ;;
    verify)    cmd_verify ;;
    archive)   cmd_archive "$@" ;;
    branch)    cmd_branch "$@" ;;
    finish)    cmd_finish ;;
    publish)   cmd_publish ;;
    status)    cmd_status ;;
    ayuda|help|-h|--help) cmd_help ;;
    *) cmd_help; die "Comando desconocido: $cmd" ;;
  esac
}

main "$@"
