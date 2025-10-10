// ignore_for_file: avoid_web_libraries_in_flutter, deprecated_member_use
import 'dart:html' as html;

/// Attempts to close the current browser tab/window. If blocked by the browser,
/// navigates to about:blank as a fallback.
void closeApp() {
  // Try to close the window (works if the tab was opened via script or allowed)
  html.window.close();
  // As a fallback, navigate away to a blank page
  html.window.location.href = 'about:blank';
}
