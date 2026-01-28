# Скрипт для копіювання коду проекту в буфер обміну або файл
# Використання: ./extract_code.ps1 -OutputFile "my_code.txt"

param(
    [string]$ProjectPath = ".",
    [string]$OutputFile = $null
)

# Розширення файлів, які потрібно включити
$CodeExtensions = @(
    '.py', '.js', '.ts', '.tsx', '.jsx', '.vue', '.svelte',
    '.html', '.css', '.scss', '.sass', '.less',
    '.php', '.rb', '.go', '.rs', '.java', '.cpp', '.c', '.h',
    '.sql', '.yml', '.yaml', '.json', '.xml', '.md'
)

# Директорії та шаблони файлів, які потрібно пропустити
$SkipPatterns = @(
    '.venv', 'venv', 'env', '.env', '.next', 'next', 'node_modules', 
    '__pycache__', '.git', '.vscode', '.idea', 'dist', 'build', 'coverage', 
    '.pytest_cache', '.mypy_cache', 'staticfiles', 'media', '.DS_Store', 
    'Thumbs.db', 'celerybeat-schedule', '*.pyc', '*.pyo', '*.pyd', 
    '*.sqlite3', '*.log', '*.tmp', '*.cache', '*.pid', '*.lock', 'processed_knowledge', 'data'
)

# Конкретні назви файлів, які потрібно пропустити
$SkipFiles = @(
    'package-lock.json', 'yarn.lock', 'composer.json', 'composer.lock',
    'Pipfile', 'Pipfile.lock', 'poetry.lock', 'setup.py', 'setup.cfg',
    'pyproject.toml', 'tox.ini', 'pytest.ini', '.gitignore', '.dockerignore',
    '.env.example', 'LICENSE', 'CHANGELOG.md', 'CONTRIBUTING.md', 'AUTHORS', 'MANIFEST.in'
)

# --- Більше не редагуй нижче ---

function Should-Skip-Path {
    param(
        [System.IO.FileInfo]$FileObject,
        [string]$RootPath
    )

    # 1. Перевірка на точну назву файлу
    if ($SkipFiles -contains $FileObject.Name) {
        return $true
    }

    # 2. Перевірка на наявність забороненої директорії у шляху
    $relativePath = $FileObject.FullName.Substring($RootPath.Length).TrimStart('\', '/')
    $pathParts = $relativePath.Split([System.IO.Path]::DirectorySeparatorChar, [System.IO.Path]::AltDirectorySeparatorChar)

    foreach ($part in $pathParts) {
        if ($SkipPatterns -contains $part) {
            # FIX 2: Закоментовано вивід логів про пропущені файли для чистоти
            # if($part -in @('.venv', 'venv', 'node_modules', '.next', 'dist', 'build')){
            #      Write-Host "SKIP: $relativePath (містить заборонену директорію '$part')" -ForegroundColor DarkGray
            # }
            return $true
        }
    }
    
    # 3. Перевірка по шаблону (wildcard), напр. "*.log"
    $wildcardPatterns = $SkipPatterns | Where-Object { $_.Contains("*") }
    foreach ($pattern in $wildcardPatterns) {
        if ($FileObject.Name -like $pattern) {
            return $true
        }
    }

    return $false
}

function Get-Code-Content {
    param([string]$RootPath)
    
    $output = [System.Text.StringBuilder]::new()
    $output.AppendLine("# СТРУКТУРА ПРОЕКТУ ТА КОД") | Out-Null
    $output.AppendLine("# Згенеровано: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')") | Out-Null
    $output.AppendLine("# Проект: $(Split-Path $RootPath -Leaf)") | Out-Null
    $output.AppendLine(("=" * 80)) | Out-Null
    $output.AppendLine() | Out-Null
    
    $allFiles = Get-ChildItem -Path $RootPath -File -Recurse | Sort-Object FullName
    
    $processedFiles = @()

    foreach ($file in $allFiles) {
        if (-not ($CodeExtensions -contains $file.Extension.ToLower())) {
            continue
        }
        
        if (Should-Skip-Path -FileObject $file -RootPath $RootPath) {
            continue
        }

        $processedFiles += $file
    }

    Write-Host "Знайдено $($processedFiles.Count) файлів з кодом" -ForegroundColor Green
    
    foreach ($file in $processedFiles) {
        try {
            $relativePath = $file.FullName.Substring($RootPath.Length).TrimStart('\', '/')
            $relativePath = $relativePath.Replace('\', '/')
            
            Write-Host "Обробляється: $relativePath" -ForegroundColor Yellow
            
            $output.AppendLine() | Out-Null
            $output.AppendLine(("=" * 50)) | Out-Null
            $output.AppendLine("ФАЙЛ: $relativePath") | Out-Null
            $output.AppendLine("РОЗМІР: $([math]::Round($file.Length / 1KB, 2)) KB") | Out-Null
            $output.AppendLine(("=" * 50)) | Out-Null
            $output.AppendLine() | Out-Null
            
            # FIX 1: Використовуємо -LiteralPath, щоб коректно обробляти шляхи з [ ]
            $content = Get-Content -LiteralPath $file.FullName -Raw -Encoding UTF8
            
            if ($content) {
                $output.AppendLine($content) | Out-Null
            } else {
                $output.AppendLine("# Файл пустий або не вдалося прочитати") | Out-Null
            }
            
            $output.AppendLine() | Out-Null
            
        } catch {
            Write-Warning "Помилка при обробці файлу $($file.FullName): $($_.Exception.Message)"
            $output.AppendLine("# ПОМИЛКА: Не вдалося прочитати файл $relativePath") | Out-Null
            $output.AppendLine() | Out-Null
        }
    }
    
    $output.AppendLine() | Out-Null
    $output.AppendLine(("=" * 80)) | Out-Null
    $output.AppendLine("# КІНЕЦЬ ФАЙЛУ") | Out-Null
    $output.AppendLine("# Оброблено файлів: $($processedFiles.Count)") | Out-Null
    $output.AppendLine(("=" * 80)) | Out-Null
    
    return $output.ToString()
}

# --- Основна логіка ---
$fullPath = Resolve-Path $ProjectPath
if (-not $fullPath.ToString().EndsWith([System.IO.Path]::DirectorySeparatorChar)) {
    $fullPath = "$fullPath" + [System.IO.Path]::DirectorySeparatorChar
}
Write-Host "Обробляється проект: $fullPath" -ForegroundColor Cyan

$codeContent = Get-Code-Content -RootPath $fullPath

if ($OutputFile) {
    Set-Content -Path $OutputFile -Value $codeContent -Encoding UTF8
    Write-Host "Код збережено у файл: $OutputFile" -ForegroundColor Green
} else {
    try {
        Set-Clipboard -Value $codeContent
        Write-Host "Код скопійовано в буфер обміну!" -ForegroundColor Green
        $contentSizeKB = [math]::Round(($codeContent.Length * 2) / 1024, 2)
        Write-Host "Розмір: $contentSizeKB KB" -ForegroundColor Cyan
    } catch {
        Write-Error "Помилка при копіюванні в буфер обміну: $($_.Exception.Message)"
        Write-Host "Спробуємо зберегти у файл..." -ForegroundColor Yellow
        $tempFile = "project_code_$(Get-Date -Format 'yyyyMMdd_HHmmss').txt"
        Set-Content -Path $tempFile -Value $codeContent -Encoding UTF8
        Write-Host "Код збережено у файл: $tempFile" -ForegroundColor Green
    }
}

Write-Host "Готово!" -ForegroundColor Green