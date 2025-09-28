# Compiler settings
NVCC = nvcc
CXX = g++

# Directories
SRCDIR = src
INCDIR = include
BINDIR = bin
BUILDDIR = build

# Compiler flags
NVCCFLAGS = -O2 -arch=sm_60 -I$(INCDIR) -allow-unsupported-compiler
CXXFLAGS = -O2 -std=c++14 -I$(INCDIR)
LDFLAGS = -lcudart

# Find all .cu files in src directory
CUDA_SOURCES = $(wildcard $(SRCDIR)/*.cu)
CUDA_TARGETS = $(patsubst $(SRCDIR)/%.cu, $(BINDIR)/%, $(CUDA_SOURCES))

# Default target
all: $(CUDA_TARGETS)

# Rule to build CUDA executables
$(BINDIR)/%: $(SRCDIR)/%.cu | $(BINDIR) $(BUILDDIR)
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LDFLAGS)

# Create directories if they don't exist
$(BINDIR):
	mkdir -p $(BINDIR)

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

# Clean target
clean:
	rm -rf $(BINDIR)/* $(BUILDDIR)/*

# Phony targets
.PHONY: all clean

# Help target
help:
	@echo "Available targets:"
	@echo "  all     - Build all CUDA programs"
	@echo "  clean   - Remove all built files"
	@echo "  help    - Show this help message"