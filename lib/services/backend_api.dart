import 'dart:convert';
import 'package:http/http.dart' as http;

/// Simple API client for the CysticCare FastAPI backend.
class BackendApi {
  /// Override at runtime with:
  /// flutter run --dart-define=BACKEND_BASE_URL=http://10.0.2.2:8001
  /// Defaults to localhost which works for iOS simulator, macOS, and web.
  static const String _rawBase = String.fromEnvironment(
    'BACKEND_BASE_URL',
    defaultValue: 'http://localhost:8001',
  );

  // Normalize base URL (trim whitespace/newlines and trailing slashes)
  static final String _baseUrl = _normalizeUrl(_rawBase);
  
  static String _normalizeUrl(String url) {
    var v = url.trim();
    // Remove any trailing slashes
    while (v.endsWith('/')) {
      v = v.substring(0, v.length - 1);
    }
    return v;
  }

  Uri _uri(String path) {
    final normalizedPath = path.startsWith('/') ? path : '/$path';
    return Uri.parse('$_baseUrl$normalizedPath');
  }

  Future<Map<String, dynamic>> health() async {
    final res = await http.get(_uri('/health'));
    if (res.statusCode >= 200 && res.statusCode < 300) {
      return jsonDecode(res.body) as Map<String, dynamic>;
    }
    throw Exception('Health failed: ${res.statusCode} ${res.body}');
  }

  Future<String> initializeSession() async {
    final res = await http.post(_uri('/initialize'));
    if (res.statusCode >= 200 && res.statusCode < 300) {
      final data = jsonDecode(res.body) as Map<String, dynamic>;
      final sessionId = data['session_id'] as String?;
      if (sessionId == null || sessionId.isEmpty) {
        throw Exception('No session_id in response');
      }
      return sessionId;
    }
    throw Exception('Initialize failed: ${res.statusCode} ${res.body}');
  }

  Future<ChatReply> chat({
    required String sessionId,
    required String message,
  }) async {
    final res = await http.post(
      _uri('/chat'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'message': message, 'session_id': sessionId}),
    );

    if (res.statusCode >= 200 && res.statusCode < 300) {
      final data = jsonDecode(res.body) as Map<String, dynamic>;
      return ChatReply(
        response: (data['response'] ?? '').toString(),
        sourceTitles: (data['source_titles'] as List<dynamic>? ?? [])
            .map((e) => e.toString())
            .toList(),
        sourceAuthors: (data['source_authors'] as List<dynamic>? ?? [])
            .map((e) => e.toString())
            .toList(),
      );
    }
    throw Exception('Chat failed: ${res.statusCode} ${res.body}');
  }
}

class ChatReply {
  final String response;
  final List<String> sourceTitles;
  final List<String> sourceAuthors;

  ChatReply({
    required this.response,
    required this.sourceTitles,
    required this.sourceAuthors,
  });
}
