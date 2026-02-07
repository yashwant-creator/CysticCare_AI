import 'package:flutter/material.dart';
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
  final String content;
  final bool isUser;
  final DateTime timestamp;

  ChatMessage({
    required this.content,
    required this.isUser,
    required this.timestamp,
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
  String? _sessionId;
  String? _error;
  bool _disclaimerAccepted = false;

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

  void _addMessage(String content, {required bool isUser}) {
    setState(() {
      _messages.add(
        ChatMessage(
          content: content,
          isUser: isUser,
          timestamp: DateTime.now(),
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
    if (!_isSessionInitialized || _sessionId == null) {
      setState(() => _error = 'Session is not initialized yet.');
      return;
    }

    _addMessage(text, isUser: true);
    _textController.clear();

    setState(() {
      _isLoading = true;
      _error = null;
    });

    try {
      final reply = await _api.chat(sessionId: _sessionId!, message: text);
      setState(() {
        _isLoading = false;
      });

      String response = reply.response;
      // Append sources/citations if available
      if (reply.sourceCitations.isNotEmpty) {
        response = '$response\n\n📚 Sources:\n';
        for (int i = 0; i < reply.sourceCitations.length; i++) {
          final citation = reply.sourceCitations[i];
          final title = i < reply.sourceTitles.length ? reply.sourceTitles[i] : '';
          response += '${i + 1}. $citation';
          if (title.isNotEmpty && !citation.contains(title)) {
            response += ' - $title';
          }
          response += '\n';
        }
      }
      _addMessage(response, isUser: false);
    } catch (e) {
      setState(() {
        _isLoading = false;
        _error = 'Failed to send message: $e';
      });
    }
  }

  // Removed mock response generator; responses now come from backend API.

  void _clearChat() {
    setState(() {
      _messages.clear();
    });
    _addMessage(
      "Chat history cleared. How can I help you with PKD-related questions?",
      isUser: false,
    );
  }

  void _resetSession() {
    setState(() {
      _messages.clear();
      _isSessionInitialized = false;
      _sessionId = null;
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
          else
          if (!_isSessionInitialized && _isLoading)
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
            child: Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: message.isUser
                    ? const Color(0xFF2E86AB)
                    : Colors.grey[100],
                borderRadius: BorderRadius.circular(16),
              ),
              child: Text(
                message.content,
                style: TextStyle(
                  color: message.isUser ? Colors.white : Colors.black87,
                  fontSize: 16,
                ),
              ),
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
            child: const Text(
              'CysticCare AI is thinking...',
              style: TextStyle(
                color: Colors.black54,
                fontStyle: FontStyle.italic,
              ),
            ),
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
              onSubmitted: _sendMessage,
            ),
          ),
          const SizedBox(width: 8),
          FloatingActionButton(
            onPressed: () => _sendMessage(_textController.text),
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
    _textController.dispose();
    _scrollController.dispose();
    super.dispose();
  }
}
