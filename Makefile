.PHONY: setup install submodule-update cpp_binding clean help submodule-clean submodule-init git-fix
PYTHON ?= python

# Setup development environment
setup:
	@echo "🚀 Setting up development environment..."
	@$(MAKE) submodule-init || echo "❌ submodule-init failed, continuing..."
	@$(MAKE) install || echo "❌ install failed, continuing..."
	@$(MAKE) cpp_binding || echo "❌ cpp_binding failed, continuing..."
	@echo "✅ Setup completed (check messages above for any failures)"


# Clean up submodule issues
submodule-clean:
	@echo "🧹 Cleaning up submodule index issues..."
	@echo "Removing from git index..."
	@git rm --cached third_party/gaussian_splatting 2>/dev/null || true
	@echo "Removing directory completely..."
	@rm -rf third_party/gaussian_splatting
	@echo "Cleaning git modules cache..."
	@rm -rf .git/modules/third_party/gaussian_splatting 2>/dev/null || true
	@rm -rf .git/modules/third_party 2>/dev/null || true
	@echo "Cleaning .gitmodules..."
	@if [ -f ".gitmodules" ]; then \
		sed -i '/\[submodule "third_party\/gaussian_splatting"\]/,+2d' .gitmodules 2>/dev/null || true; \
		if [ ! -s ".gitmodules" ]; then rm -f .gitmodules; fi; \
	fi
	@echo "Cleaning git config..."
	@git config --remove-section submodule.third_party/gaussian_splatting 2>/dev/null || true

# Fix git repository corruption
git-fix:
	@echo "🔧 Fixing git repository corruption..."
	@echo "Running git fsck..."
	@git fsck --full 2>/dev/null || true
	@echo "Running git gc..."
	@git gc --aggressive --prune=now 2>/dev/null || true
	@echo "Git repair completed"

# Initialize third-party dependencies (NO submodules - direct clone)
submodule-init: submodule-clean git-fix
	@echo "📦 Initializing third-party dependencies..."
	@mkdir -p third_party
	@echo "Cloning gaussian-splatting directly..."
	@git clone https://github.com/graphdeco-inria/gaussian-splatting.git third_party/gaussian_splatting
	@cd third_party/gaussian_splatting && git checkout main
	@echo "✅ Successfully cloned gaussian-splatting"

# Install dependencies
install:
	@echo "📋 Installing dependencies..."
	@if [ -f "environment.yaml" ]; then \
		echo "Found environment.yaml, using conda..."; \
		conda env update -f environment.yaml; \
	else \
		echo "No dependency file found (environment.yaml)"; \
		exit 1; \
	fi

# Update git submodules (for existing submodules only)
submodule-update:
	@echo "🔄 Updating submodules..."
	git submodule update --init --recursive
	git submodule foreach git pull origin main

# Build C++ bindings
cpp_binding:
	@echo "🔧 Building C++ bindings..."
	$(PYTHON) setup.py build_ext --inplace

# Clean build artifacts
clean:
	@echo "🧹 Cleaning build artifacts..."
	$(PYTHON) setup.py clean --all
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -name "*.so" -delete
	find . -name "*.pyd" -delete
	find . -path "./navisim/*.so" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete

# Rebuild everything
rebuild: clean cpp_binding

# Show help
help:
	@echo "Available targets:"
	@echo "  setup            - Complete development setup (first time)"
	@echo "  git-fix          - Fix git repository corruption"
	@echo "  submodule-clean  - Clean up submodule index issues"
	@echo "  submodule-init   - Initialize third-party dependencies"
	@echo "  install          - Install dependencies (conda env or pip)"
	@echo "  submodule-update - Update existing submodules"
	@echo "  cpp_binding      - Build C++ bindings"
	@echo "  clean            - Remove build artifacts"
	@echo "  rebuild          - Clean and rebuild C++ bindings"
	@echo "  help             - Show this help message"
	@echo ""
	@echo "Dependency files (checked in order):"
	@echo "  environment.yaml - Conda environment file"
	@echo "  environment.yml  - Conda environment file (alternative)"
	@echo "  requirements.txt - Pip requirements file"