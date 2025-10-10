import 'package:flutter/services.dart';

/// Closes the app on mobile/desktop platforms.
/// Uses SystemNavigator.pop which is the recommended way to programmatically
/// close the app on Android. On iOS, programmatic exit is discouraged, but
/// this will attempt to pop the app to the background.
void closeApp() {
  SystemNavigator.pop();
}
