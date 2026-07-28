import 'package:flutter/material.dart';
import 'package:flutter_markdown_plus/flutter_markdown_plus.dart';
import 'services/backend_api.dart';
import 'utils/close_app.dart';

void main() {
  runApp(const CysticCareApp());
}

class CysticCareApp extends StatelessWidget {
  const CysticCareApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'CysticCare AI - AI Support for Polycystic Kidney Disease',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        primaryColor: const Color(0xFF2E86AB),
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF2E86AB),
          brightness: Brightness.light,
        ),
        useMaterial3: true,
      ),
      home: const ChatScreen(),
      debugShowCheckedModeBanner: false,
    );
  }
}

class ChatMessage {
  String content;
  final bool isUser;
  final DateTime timestamp;
  ValidationInfo? validation;

  ChatMessage({
    required this.content,
    required this.isUser,
    required this.timestamp,
    this.validation,
  });
}

class ChatScreen extends StatefulWidget {
  const ChatScreen({super.key});

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  final List<ChatMessage> _messages = [];
  final TextEditingController _textController = TextEditingController();
  final ScrollController _scrollController = ScrollController();
  final BackendApi _api = BackendApi();
  bool _isSessionInitialized = false;
  bool _isLoading = false;
  bool _isRequestActive = false;
  int _requestEpoch = 0;
  String? _currentStatus;
  String? _sessionId;
  String? _error;
  bool _disclaimerAccepted = false;
  List<String> _followupQuestions = [];

  static const List<String> quickQuestions = [
    "What is Polycystic Kidney Disease?",
    "What are the symptoms of PKD?",
    "How is PKD diagnosed?",
    "What treatment options are available?",
    "How can I manage PKD symptoms?",
    "What lifestyle changes can help with PKD?",
  ];

  @override
  void initState() {
    super.initState();
    // Show disclaimer as soon as the first frame is rendered
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _showDisclaimerIfNeeded();
    });
  }

  Future<void> _showDisclaimerIfNeeded() async {
    if (_disclaimerAccepted) return;

    final accepted = await showDialog<bool>(
      context: context,
      barrierDismissible: false,
      builder: (ctx) {
        return AlertDialog(
          title: const Text('Disclaimer'),
          content: const SingleChildScrollView(
            child: Text(
              'The information contained in this website is not intended to serve as a replacement for professional medical advice. Any use of the information in this website is at the reader\'s discretion. The author and publisher specifically disclaim any and all liability arising directly or indirectly from the use or application of any information contained in this website. A health care professional should be consulted regarding your specific situation.',
            ),
          ),
          actions: [
            TextButton(
              onPressed: () {
                Navigator.of(ctx).pop(false);
              },
              child: const Text('Decline'),
            ),
            FilledButton(
              onPressed: () {
                Navigator.of(ctx).pop(true);
              },
              child: const Text('Accept'),
            ),
          ],
        );
      },
    );

    if (accepted == true) {
      setState(() => _disclaimerAccepted = true);
      _initializeSession();
    } else {
      // Close the website/app
      closeApp();
    }
  }

  Future<void> _initializeSession() async {
    setState(() {
      _isLoading = true;
      _error = null;
    });

    try {
      // Optional health check
      await _api.health();
      final id = await _api.initializeSession();
      setState(() {
        _sessionId = id;
        _isSessionInitialized = true;
        _isLoading = false;
      });

      // Add welcome message
      _addMessage(
        "Welcome to CysticCare AI! I'm connected to the knowledge base on PKD and ready to help. What would you like to know?",
        isUser: false,
      );
    } catch (e) {
      setState(() {
        _isLoading = false;
        _error =
            'Failed to initialize session. Please ensure the backend is running. Error: $e';
      });
    }
  }

  void _addMessage(
    String content, {
    required bool isUser,
    ValidationInfo? validation,
  }) {
    setState(() {
      _messages.add(
        ChatMessage(
          content: content,
          isUser: isUser,
          timestamp: DateTime.now(),
          validation: validation,
        ),
      );
    });
    _scrollToBottom();
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      }
    });
  }

  Future<void> _sendMessage(String text) async {
    if (text.trim().isEmpty) return;
    if (_isRequestActive) return;
    if (!_isSessionInitialized || _sessionId == null) {
      setState(() => _error = 'Session is not initialized yet.');
      return;
    }

    _addMessage(text, isUser: true);
    _textController.clear();

    setState(() {
      _isLoading = true;
      _isRequestActive = true;
      _currentStatus = "Connecting...";
      _error = null;
      _followupQuestions = [];
    });

    ChatMessage? activeBotMessage;
    List<String> sourceTitles = [];
    List<String> sourceCitations = [];

    final requestEpoch = ++_requestEpoch;
    try {
      final stream = _api.chatStream(sessionId: _sessionId!, message: text);

      await for (final event in stream) {
        if (requestEpoch != _requestEpoch) break;
        switch (event.type) {
          case ChatStreamEventType.status:
            setState(() {
              _currentStatus = event.status;
              // Keep typing indicator visible for post-generation processes
              if (activeBotMessage != null) {
                _isLoading = true;
              }
            });
            _scrollToBottom();
            break;

          case ChatStreamEventType.sources:
            if (event.sourceTitles != null) {
              sourceTitles = event.sourceTitles!;
            }
            if (event.sourceCitations != null) {
              sourceCitations = event.sourceCitations!;
            }
            break;

          case ChatStreamEventType.chunk:
            if (event.chunk == null) break;
            if (activeBotMessage == null) {
              activeBotMessage = ChatMessage(
                content: event.chunk!,
                isUser: false,
                timestamp: DateTime.now(),
              );
              setState(() {
                _isLoading =
                    false; // Hide typing indicator and let bubble render
                _messages.add(activeBotMessage!);
              });
            } else {
              setState(() {
                activeBotMessage!.content += event.chunk!;
              });
            }
            _scrollToBottom();
            break;

          case ChatStreamEventType.followup:
            setState(() {
              if (event.followupQuestions != null) {
                _followupQuestions = event.followupQuestions!;
              }
            });
            break;

          case ChatStreamEventType.validation:
            if (event.validation == null) break;
            setState(() {
              if (activeBotMessage != null) {
                if (event.validationText != null) {
                  // Handle corrective regeneration: smooth swap/animation
                  activeBotMessage.content = event.validationText!;
                }
                activeBotMessage.validation = event.validation;
              }
            });
            _scrollToBottom();
            break;

          case ChatStreamEventType.error:
            if (event.error != null) {
              setState(() {
                _error = 'Stream Error: ${event.error}';
              });
            }
            break;

          case ChatStreamEventType.done:
            break;
        }
      }

      if (requestEpoch != _requestEpoch) return;

      // After stream successfully completes, append any buffered sources
      if (activeBotMessage != null && sourceCitations.isNotEmpty) {
        String sourcesBlock = '\n\n━━━━━━━━━━━━━━━━\n📚 Sources:\n\n';
        for (int i = 0; i < sourceCitations.length; i++) {
          final authorYear = sourceCitations[i];
          final title = i < sourceTitles.length ? sourceTitles[i] : '';

          sourcesBlock += '[${i + 1}] $authorYear';
          if (title.isNotEmpty && title != 'Unknown') {
            sourcesBlock += ':\n    "$title"';
          }
          sourcesBlock += '\n\n';
        }
        setState(() {
          activeBotMessage!.content += sourcesBlock;
        });
        _scrollToBottom();
      }

      setState(() {
        _isLoading = false;
        _currentStatus = null;
      });
    } catch (e) {
      if (mounted && requestEpoch == _requestEpoch) {
        setState(() {
          _isLoading = false;
          _currentStatus = null;
          _error = 'Failed to stream response: $e';
        });
      }
    } finally {
      if (mounted && requestEpoch == _requestEpoch) {
        setState(() {
          _isRequestActive = false;
          _isLoading = false;
          _currentStatus = null;
        });
      }
    }
  }

  // Removed mock response generator; responses now come from backend API.

  void _clearChat() {
    _requestEpoch++;
    setState(() {
      _messages.clear();
      _isLoading = false;
      _isRequestActive = false;
      _currentStatus = null;
    });
    _addMessage(
      "Chat history cleared. How can I help you with PKD-related questions?",
      isUser: false,
    );
  }

  void _resetSession() {
    _requestEpoch++;
    setState(() {
      _messages.clear();
      _isSessionInitialized = false;
      _sessionId = null;
      _isLoading = false;
      _isRequestActive = false;
      _currentStatus = null;
    });
    _initializeSession();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Row(
          children: [
            // CysticCare AI Logo
            Container(
              width: 40,
              height: 40,
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(20),
              ),
              child: ClipRRect(
                borderRadius: BorderRadius.circular(20),
                child: Image.asset(
                  'assets/images/logo.png',
                  fit: BoxFit.cover,
                  errorBuilder: (context, error, stackTrace) {
                    // Fallback to icon if logo fails to load
                    return Container(
                      decoration: BoxDecoration(
                        color: Theme.of(context).primaryColor,
                        borderRadius: BorderRadius.circular(20),
                      ),
                      child: const Icon(
                        Icons.medical_services,
                        color: Colors.white,
                        size: 24,
                      ),
                    );
                  },
                ),
              ),
            ),
            const SizedBox(width: 12),
            const Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'CysticCare AI',
                  style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
                ),
                Text(
                  'AI Support for PKD',
                  style: TextStyle(fontSize: 12, fontWeight: FontWeight.normal),
                ),
              ],
            ),
          ],
        ),
        backgroundColor: Colors.white,
        foregroundColor: const Color(0xFF2E86AB),
        elevation: 1,
      ),
      drawer: _buildDrawer(),
      body: Column(
        children: [
          if (_error != null)
            Container(
              width: double.infinity,
              color: Colors.red.shade50,
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              child: Row(
                children: [
                  const Icon(Icons.error_outline, color: Colors.red),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      _error!,
                      style: const TextStyle(color: Colors.red),
                    ),
                  ),
                  IconButton(
                    icon: const Icon(Icons.close, color: Colors.red),
                    onPressed: () => setState(() => _error = null),
                  ),
                ],
              ),
            ),
          _buildAboutSection(),
          if (!_disclaimerAccepted)
            const Expanded(
              child: Center(
                child: Text('Please accept the disclaimer to continue.'),
              ),
            )
          else if (!_isSessionInitialized && _isLoading)
            const Expanded(
              child: Center(
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    CircularProgressIndicator(),
                    SizedBox(height: 16),
                    Text('Initializing CysticCare AI session...'),
                  ],
                ),
              ),
            )
          else
            Expanded(child: _buildChatArea()),
          if (_followupQuestions.isNotEmpty && !_isLoading)
            _buildFollowUpSuggestions(),
        ],
      ),
    );
  }

  Widget _buildAboutSection() {
    return Container(
      margin: const EdgeInsets.all(16),
      child: ExpansionTile(
        leading: const Icon(Icons.info_outline, color: Color(0xFF2E86AB)),
        title: const Text(
          'About CysticCare AI',
          style: TextStyle(
            fontWeight: FontWeight.bold,
            color: Color(0xFF2E86AB),
          ),
        ),
        initiallyExpanded: false,
        children: [
          Container(
            padding: const EdgeInsets.all(16),
            child: const Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'CysticCare AI is an AI-powered support agent designed to help patients with Polycystic Kidney Disease (PKD).',
                  style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                ),
                SizedBox(height: 12),
                Text(
                  'What you can ask:',
                  style: TextStyle(fontWeight: FontWeight.bold),
                ),
                Text('• Questions about PKD symptoms and management'),
                Text('• Treatment options and lifestyle recommendations'),
                Text('• Support and guidance for living with PKD'),
                Text('• General information about kidney health'),
                SizedBox(height: 12),
                Text(
                  'Important: This AI assistant provides general information and support. Always consult with your healthcare provider for medical advice and treatment decisions.',
                  style: TextStyle(
                    fontStyle: FontStyle.italic,
                    color: Colors.red,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildChatArea() {
    return Column(
      children: [
        Expanded(
          child: Container(
            padding: const EdgeInsets.symmetric(horizontal: 16),
            child: ListView.builder(
              controller: _scrollController,
              itemCount: _messages.length + (_isLoading ? 1 : 0),
              itemBuilder: (context, index) {
                if (index == _messages.length && _isLoading) {
                  return _buildTypingIndicator();
                }
                return _buildMessageBubble(_messages[index]);
              },
            ),
          ),
        ),
        _buildMessageInput(),
      ],
    );
  }

  Widget _buildMessageBubble(ChatMessage message) {
    return Container(
      margin: const EdgeInsets.symmetric(vertical: 8),
      child: Row(
        mainAxisAlignment: message.isUser
            ? MainAxisAlignment.end
            : MainAxisAlignment.start,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          if (!message.isUser) ...[
            CircleAvatar(
              radius: 20,
              backgroundColor: Colors.transparent,
              child: ClipOval(
                child: Image.asset(
                  'assets/images/logo.png',
                  width: 40,
                  height: 40,
                  fit: BoxFit.cover,
                  errorBuilder: (context, error, stackTrace) {
                    // Fallback to icon with blue background if logo fails to load
                    return CircleAvatar(
                      backgroundColor: const Color(0xFF2E86AB),
                      radius: 20,
                      child: const Icon(
                        Icons.smart_toy,
                        color: Colors.white,
                        size: 20,
                      ),
                    );
                  },
                ),
              ),
            ),
            const SizedBox(width: 8),
          ],
          Flexible(
            child: Column(
              crossAxisAlignment: message.isUser
                  ? CrossAxisAlignment.end
                  : CrossAxisAlignment.start,
              children: [
                Container(
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    color: message.isUser
                        ? const Color(0xFF2E86AB)
                        : Colors.grey[100],
                    borderRadius: BorderRadius.circular(16),
                  ),
                  child: message.isUser
                      ? Text(
                          message.content,
                          style: const TextStyle(
                            color: Colors.white,
                            fontSize: 16,
                            height: 1.45,
                          ),
                        )
                      : MarkdownBody(
                          data: message.content,
                          selectable: true,
                          styleSheet:
                              MarkdownStyleSheet.fromTheme(
                                Theme.of(context),
                              ).copyWith(
                                p: const TextStyle(
                                  color: Colors.black87,
                                  fontSize: 16,
                                  height: 1.5,
                                ),
                                strong: const TextStyle(
                                  color: Colors.black87,
                                  fontSize: 16,
                                  fontWeight: FontWeight.w700,
                                  height: 1.5,
                                ),
                                em: const TextStyle(
                                  color: Colors.black87,
                                  fontSize: 16,
                                  fontStyle: FontStyle.italic,
                                  height: 1.5,
                                ),
                                h1: const TextStyle(
                                  color: Colors.black87,
                                  fontSize: 22,
                                  fontWeight: FontWeight.w700,
                                  height: 1.35,
                                ),
                                h2: const TextStyle(
                                  color: Colors.black87,
                                  fontSize: 20,
                                  fontWeight: FontWeight.w700,
                                  height: 1.35,
                                ),
                                h3: const TextStyle(
                                  color: Colors.black87,
                                  fontSize: 18,
                                  fontWeight: FontWeight.w700,
                                  height: 1.4,
                                ),
                                listBullet: const TextStyle(
                                  color: Colors.black87,
                                  fontSize: 16,
                                  height: 1.5,
                                ),
                                blockSpacing: 12,
                                listIndent: 22,
                              ),
                        ),
                ),
                if (!message.isUser && message.validation != null)
                  _buildValidationBadge(message.validation!),
              ],
            ),
          ),
          if (message.isUser) ...[
            const SizedBox(width: 8),
            CircleAvatar(
              backgroundColor: Colors.grey[300],
              child: const Icon(Icons.person, color: Colors.black54, size: 20),
            ),
          ],
        ],
      ),
    );
  }

  Widget _buildValidationBadge(ValidationInfo validation) {
    final score = (validation.overallScore * 100).round();
    final Color badgeColor;
    final IconData badgeIcon;
    final String label;

    if (validation.passed) {
      badgeColor = const Color(0xFF4CAF50); // green
      badgeIcon = Icons.verified;
      label = 'Validated $score%';
    } else if (validation.overallScore >= 0.5) {
      badgeColor = const Color(0xFFFFA726); // orange
      badgeIcon = Icons.warning_amber_rounded;
      label = 'Caution $score%';
    } else {
      badgeColor = const Color(0xFFEF5350); // red
      badgeIcon = Icons.error_outline;
      label = 'Low confidence $score%';
    }

    return Padding(
      padding: const EdgeInsets.only(top: 6),
      child: GestureDetector(
        onTap: () => _showValidationDetails(validation),
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
          decoration: BoxDecoration(
            color: badgeColor.withValues(alpha: 0.12),
            borderRadius: BorderRadius.circular(12),
            border: Border.all(color: badgeColor.withValues(alpha: 0.3)),
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(badgeIcon, size: 14, color: badgeColor),
              const SizedBox(width: 4),
              Text(
                label,
                style: TextStyle(
                  fontSize: 12,
                  fontWeight: FontWeight.w600,
                  color: badgeColor,
                ),
              ),
              if (validation.wasRegenerated) ...[
                const SizedBox(width: 4),
                Icon(Icons.autorenew, size: 12, color: badgeColor),
              ],
            ],
          ),
        ),
      ),
    );
  }

  void _showValidationDetails(ValidationInfo validation) {
    showModalBottomSheet(
      context: context,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (ctx) {
        return Padding(
          padding: const EdgeInsets.all(20),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Icon(
                    validation.passed
                        ? Icons.verified
                        : Icons.warning_amber_rounded,
                    color: validation.passed
                        ? const Color(0xFF4CAF50)
                        : const Color(0xFFFFA726),
                    size: 24,
                  ),
                  const SizedBox(width: 8),
                  Text(
                    'Answer Validation — ${(validation.overallScore * 100).round()}%',
                    style: const TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ],
              ),
              if (validation.wasRegenerated)
                const Padding(
                  padding: EdgeInsets.only(top: 8),
                  child: Text(
                    'This answer was automatically improved by the validation agent.',
                    style: TextStyle(
                      fontSize: 13,
                      fontStyle: FontStyle.italic,
                      color: Colors.black54,
                    ),
                  ),
                ),
              const SizedBox(height: 16),
              ...validation.checks.entries.map((entry) {
                final name = entry.key;
                final check = entry.value;
                final pct = (check.score * 100).round();
                return Padding(
                  padding: const EdgeInsets.symmetric(vertical: 4),
                  child: Row(
                    children: [
                      Icon(
                        check.passed ? Icons.check_circle : Icons.cancel,
                        size: 18,
                        color: check.passed
                            ? const Color(0xFF4CAF50)
                            : const Color(0xFFEF5350),
                      ),
                      const SizedBox(width: 8),
                      Expanded(
                        child: Text(
                          _formatCheckName(name),
                          style: const TextStyle(fontSize: 14),
                        ),
                      ),
                      Text(
                        '$pct%',
                        style: TextStyle(
                          fontSize: 14,
                          fontWeight: FontWeight.w600,
                          color: check.passed
                              ? const Color(0xFF4CAF50)
                              : const Color(0xFFEF5350),
                        ),
                      ),
                    ],
                  ),
                );
              }),
              if (validation.warnings.isNotEmpty) ...[
                const SizedBox(height: 12),
                const Text(
                  'Warnings:',
                  style: TextStyle(
                    fontSize: 13,
                    fontWeight: FontWeight.w600,
                    color: Colors.black54,
                  ),
                ),
                const SizedBox(height: 4),
                ...validation.warnings.map(
                  (w) => Padding(
                    padding: const EdgeInsets.only(left: 8, top: 2),
                    child: Text(
                      '• $w',
                      style: const TextStyle(
                        fontSize: 12,
                        color: Colors.black54,
                      ),
                    ),
                  ),
                ),
              ],
              const SizedBox(height: 16),
            ],
          ),
        );
      },
    );
  }

  String _formatCheckName(String key) {
    switch (key) {
      case 'relevance':
        return 'Relevance';
      case 'source_attribution':
        return 'Source Attribution';
      case 'safety':
        return 'Safety & Disclaimers';
      default:
        return key;
    }
  }

  Widget _buildTypingIndicator() {
    return Container(
      margin: const EdgeInsets.symmetric(vertical: 8),
      child: Row(
        children: [
          CircleAvatar(
            radius: 20,
            backgroundColor: Colors.transparent,
            child: ClipOval(
              child: Image.asset(
                'assets/images/logo.png',
                width: 40,
                height: 40,
                fit: BoxFit.cover,
                errorBuilder: (context, error, stackTrace) {
                  // Fallback to icon with blue background if logo fails to load
                  return CircleAvatar(
                    backgroundColor: const Color(0xFF2E86AB),
                    radius: 20,
                    child: const Icon(
                      Icons.smart_toy,
                      color: Colors.white,
                      size: 20,
                    ),
                  );
                },
              ),
            ),
          ),
          const SizedBox(width: 8),
          Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: Colors.grey[100],
              borderRadius: BorderRadius.circular(16),
            ),
            child: Text(
              _currentStatus ?? 'CysticCare AI is thinking...',
              style: const TextStyle(
                color: Colors.black54,
                fontStyle: FontStyle.italic,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildFollowUpSuggestions() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
      decoration: BoxDecoration(
        color: Colors.grey[50],
        border: Border(top: BorderSide(color: Colors.grey[200]!)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'Suggested follow-up questions:',
            style: TextStyle(
              fontSize: 13,
              fontWeight: FontWeight.w600,
              color: Color(0xFF2E86AB),
            ),
          ),
          const SizedBox(height: 8),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: _followupQuestions.map((question) {
              return ActionChip(
                label: Text(question, style: const TextStyle(fontSize: 13)),
                backgroundColor: Colors.white,
                side: const BorderSide(color: Color(0xFF2E86AB)),
                labelStyle: const TextStyle(color: Color(0xFF2E86AB)),
                onPressed: () => _sendMessage(question),
              );
            }).toList(),
          ),
        ],
      ),
    );
  }

  Widget _buildMessageInput() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        boxShadow: [
          BoxShadow(
            color: Colors.grey.withValues(alpha: 0.2),
            spreadRadius: 1,
            blurRadius: 5,
            offset: const Offset(0, -2),
          ),
        ],
      ),
      child: Row(
        children: [
          Expanded(
            child: TextField(
              controller: _textController,
              decoration: InputDecoration(
                hintText:
                    'Ask CysticCare AI about Polycystic Kidney Disease...',
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(25),
                  borderSide: BorderSide(color: Colors.grey[300]!),
                ),
                focusedBorder: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(25),
                  borderSide: const BorderSide(color: Color(0xFF2E86AB)),
                ),
                contentPadding: const EdgeInsets.symmetric(
                  horizontal: 20,
                  vertical: 12,
                ),
              ),
              maxLines: null,
              textInputAction: TextInputAction.send,
              onSubmitted: _isRequestActive ? null : _sendMessage,
            ),
          ),
          const SizedBox(width: 8),
          FloatingActionButton(
            onPressed: _isRequestActive
                ? null
                : () => _sendMessage(_textController.text),
            backgroundColor: const Color(0xFF2E86AB),
            child: const Icon(Icons.send, color: Colors.white),
          ),
        ],
      ),
    );
  }

  Widget _buildDrawer() {
    return Drawer(
      child: ListView(
        padding: EdgeInsets.zero,
        children: [
          DrawerHeader(
            decoration: const BoxDecoration(color: Color(0xFF2E86AB)),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Container(
                  width: 48,
                  height: 48,
                  decoration: BoxDecoration(
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: ClipRRect(
                    borderRadius: BorderRadius.circular(12),
                    child: Image.asset(
                      'assets/images/logo.png',
                      fit: BoxFit.cover,
                      errorBuilder: (context, error, stackTrace) {
                        // Fallback to icon if logo fails to load
                        return const Icon(
                          Icons.medical_services,
                          color: Colors.white,
                          size: 48,
                        );
                      },
                    ),
                  ),
                ),
                const SizedBox(height: 16),
                const Text(
                  'Options',
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 24,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
          ),
          ListTile(
            leading: const Icon(Icons.delete_outline),
            title: const Text('Clear Chat History'),
            onTap: () {
              Navigator.pop(context);
              _clearChat();
            },
          ),
          ListTile(
            leading: const Icon(Icons.refresh),
            title: const Text('Reset Session'),
            onTap: () {
              Navigator.pop(context);
              _resetSession();
            },
          ),
          const Divider(),
          const Padding(
            padding: EdgeInsets.all(16),
            child: Text(
              'Quick Questions',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
          ),
          ...quickQuestions.map(
            (question) => ListTile(
              title: Text(question, style: const TextStyle(fontSize: 14)),
              onTap: () {
                Navigator.pop(context);
                _sendMessage(question);
              },
            ),
          ),
          const Divider(),
          const Padding(
            padding: EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  '⚠️ Disclaimer',
                  style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                ),
                SizedBox(height: 8),
                Text(
                  'This AI assistant provides general information only. Always consult healthcare professionals for medical advice.',
                  style: TextStyle(fontSize: 12, color: Colors.grey),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  @override
  void dispose() {
    _requestEpoch++;
    _textController.dispose();
    _scrollController.dispose();
    super.dispose();
  }
}
