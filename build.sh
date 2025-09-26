#!/bin/bash

# Flutter Web Build Script for Netlify
set -e

echo "🚀 Starting Flutter Web build process..."

# Install Flutter if not present
if ! command -v flutter &> /dev/null; then
    echo "📦 Installing Flutter SDK..."
    git clone https://github.com/flutter/flutter.git -b stable --depth 1
    export PATH="$PATH:`pwd`/flutter/bin"
    echo "✅ Flutter SDK installed"
fi

# Enable web support
echo "🌐 Enabling Flutter web support..."
flutter config --enable-web

# Verify Flutter doctor
echo "🔍 Running Flutter doctor..."
flutter doctor --verbose

# Get dependencies
echo "📚 Getting Flutter dependencies..."
flutter pub get

# Clean previous builds
echo "🧹 Cleaning previous builds..."
flutter clean

# Build for web
echo "🏗️ Building Flutter web app..."
flutter build web --release --web-renderer html

echo "✅ Flutter web build completed successfully!"
echo "📁 Build output is in: build/web"