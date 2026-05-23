import 'dart:convert';
import 'package:http/http.dart' as http;

/// Simple API client for the CysticCare FastAPI backend.
class BackendApi {
  /// Override at runtime with:
  /// flutter run --dart-define=BACKEND_BASE_URL=http://10.0.2.2:8001
  /// Defaults to production Cloud Run backend
  static const String _rawBase = String.fromEnvironment(
    'BACKEND_BASE_URL',
    defaultValue: 'https://fastapi-backend-cot-798621865464.us-central1.run.app',
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
    // RAG system is pre-initialized with baked-in vectors, just generate a session ID
    // Session IDs are just used client-side for tracking conversations
    return 'session_${DateTime.now().millisecondsSinceEpoch}';
  }

  Future<ChatReply> chat({
    required String sessionId,
    required String message,
    bool useStepback = false,
    bool useCoT = true,
  }) async {
    final res = await http.post(
      _uri('/chat'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'query': '$message for ADPKD',
        'session_id': sessionId,
        'use_stepback': useStepback,
        'use_cot': useCoT,
        'pre_check_topic': true,
      }),
    );

    if (res.statusCode >= 200 && res.statusCode < 300) {
      final data = jsonDecode(res.body) as Map<String, dynamic>;
      
      // Parse sources array from backend
      final sources = (data['sources'] as List<dynamic>? ?? []);
      final sourceTitles = <String>[];
      final sourceAuthors = <String>[];
      final sourceCitations = <String>[];
      
      for (var source in sources) {
        if (source is Map<String, dynamic>) {
          final author = source['author']?.toString() ?? 'Unknown';
          final year = source['year']?.toString() ?? '';
          final title = source['title']?.toString() ?? '';
          
          // Create consistent, clean formatting
          final authorYear = year.isNotEmpty ? '$author ($year)' : author;
          
          sourceTitles.add(title);
          sourceAuthors.add(authorYear);
          sourceCitations.add(authorYear);
        }
      }
      
      // Parse follow-up questions
      final followupQuestions = <String>[];
      if (data['followup_questions'] is List) {
        for (var q in data['followup_questions']) {
          followupQuestions.add(q.toString());
        }
      }

      // Parse validation info
      ValidationInfo? validationInfo;
      if (data['validation'] is Map<String, dynamic>) {
        final valData = data['validation'] as Map<String, dynamic>;
        final checksRaw = valData['checks'] as Map<String, dynamic>? ?? {};
        final checks = <String, ValidationCheckResult>{};
        checksRaw.forEach((key, value) {
          if (value is Map<String, dynamic>) {
            checks[key] = ValidationCheckResult(
              passed: value['passed'] as bool? ?? true,
              score: (value['score'] as num?)?.toDouble() ?? 0.0,
            );
          }
        });
        final warnings = <String>[];
        if (valData['warnings'] is List) {
          for (var w in valData['warnings']) {
            warnings.add(w.toString());
          }
        }
        validationInfo = ValidationInfo(
          passed: valData['passed'] as bool? ?? true,
          overallScore: (valData['overall_score'] as num?)?.toDouble() ?? 0.0,
          checks: checks,
          warnings: warnings,
          wasRegenerated: valData['was_regenerated'] as bool? ?? false,
        );
      }

      return ChatReply(
        response: (data['response'] ?? '').toString(),
        sourceTitles: sourceTitles,
        sourceAuthors: sourceAuthors,
        sourceCitations: sourceCitations,
        stepbackQuery: data['stepback_query']?.toString(),
        followupQuestions: followupQuestions,
        validation: validationInfo,
      );
    }
    throw Exception('Chat failed: ${res.statusCode} ${res.body}');
  }
}

class ChatReply {
  final String response;
  final List<String> sourceTitles;
  final List<String> sourceAuthors;
  final List<String> sourceCitations;
  final String? stepbackQuery;
  final List<String> followupQuestions;
  final ValidationInfo? validation;

  ChatReply({
    required this.response,
    required this.sourceTitles,
    required this.sourceAuthors,
    required this.sourceCitations,
    this.stepbackQuery,
    this.followupQuestions = const [],
    this.validation,
  });
}

class ValidationCheckResult {
  final bool passed;
  final double score;

  ValidationCheckResult({required this.passed, required this.score});
}

class ValidationInfo {
  final bool passed;
  final double overallScore;
  final Map<String, ValidationCheckResult> checks;
  final List<String> warnings;
  final bool wasRegenerated;

  ValidationInfo({
    required this.passed,
    required this.overallScore,
    required this.checks,
    this.warnings = const [],
    this.wasRegenerated = false,
  });
}
