import 'package:flutter/material.dart';

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
  bool _isSessionInitialized = false;
  bool _isLoading = false;

  static const List<String> quickQuestions = [
    "What is Polycystic Kidney Disease?",
    "What are the symptoms of PKD?",
    "How is PKD diagnosed?",
    "What treatment options are available?",
    "How can I manage PKD symptoms?",
    "What lifestyle changes can help with PKD?"
  ];

  @override
  void initState() {
    super.initState();
    _initializeSession();
  }

  Future<void> _initializeSession() async {
    setState(() {
      _isLoading = true;
    });

    // Simulate session initialization
    await Future.delayed(const Duration(seconds: 2));
    
    setState(() {
      _isSessionInitialized = true;
      _isLoading = false;
    });

    // Add welcome message
    _addMessage(
      "Welcome to CysticCare AI! I'm here to help you with questions about Polycystic Kidney Disease. How can I assist you today?",
      isUser: false,
    );
  }

  void _addMessage(String content, {required bool isUser}) {
    setState(() {
      _messages.add(ChatMessage(
        content: content,
        isUser: isUser,
        timestamp: DateTime.now(),
      ));
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

    _addMessage(text, isUser: true);
    _textController.clear();

    setState(() {
      _isLoading = true;
    });

    // Simulate AI response delay
    await Future.delayed(const Duration(seconds: 2));
    
    // Simulate AI response (placeholder - you would integrate with your actual AI service here)
    String response = _generateMockResponse(text);
    
    setState(() {
      _isLoading = false;
    });

    _addMessage(response, isUser: false);
  }

  String _generateMockResponse(String question) {
    // Mock responses based on common PKD questions
    if (question.toLowerCase().contains('what is') && question.toLowerCase().contains('pkd')) {
      return "Polycystic Kidney Disease (PKD) is a genetic disorder characterized by the growth of numerous cysts in the kidneys. These fluid-filled cysts can gradually enlarge the kidneys and reduce their function over time.\n\n*Sources: Medical literature and PKD research*";
    } else if (question.toLowerCase().contains('symptoms')) {
      return "Common symptoms of PKD include:\n• High blood pressure\n• Pain in the back or sides\n• Blood in urine\n• Frequent urination\n• Kidney stones\n• Headaches\n\nIt's important to consult with your healthcare provider for proper evaluation and management.\n\n*Sources: PKD Foundation, Medical journals*";
    } else if (question.toLowerCase().contains('treatment')) {
      return "Treatment for PKD focuses on managing symptoms and slowing disease progression:\n• Blood pressure control\n• Pain management\n• Treatment of kidney stones\n• Management of infections\n• In advanced cases, dialysis or kidney transplant\n\nAlways work with your healthcare team to develop the best treatment plan for your specific situation.\n\n*Sources: Clinical guidelines, Medical research*";
    } else {
      return "Thank you for your question about PKD. I'm here to provide general information and support. For specific medical advice, please consult with your healthcare provider. Could you please provide more details about what you'd like to know?\n\n*Sources: General medical knowledge*";
    }
  }

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
    });
    _initializeSession();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Row(
          children: [
            // Placeholder for logo - you can add your actual logo here
            Container(
              width: 40,
              height: 40,
              decoration: BoxDecoration(
                color: Theme.of(context).primaryColor,
                borderRadius: BorderRadius.circular(20),
              ),
              child: const Icon(
                Icons.medical_services,
                color: Colors.white,
                size: 24,
              ),
            ),
            const SizedBox(width: 12),
            const Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'CysticCare AI',
                  style: TextStyle(
                    fontSize: 20,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                Text(
                  'AI Support for PKD',
                  style: TextStyle(
                    fontSize: 12,
                    fontWeight: FontWeight.normal,
                  ),
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
          _buildAboutSection(),
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
            Expanded(
              child: _buildChatArea(),
            ),
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
                Text('What you can ask:', style: TextStyle(fontWeight: FontWeight.bold)),
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
        mainAxisAlignment: message.isUser ? MainAxisAlignment.end : MainAxisAlignment.start,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          if (!message.isUser) ...[
            CircleAvatar(
              backgroundColor: const Color(0xFF2E86AB),
              child: const Icon(Icons.smart_toy, color: Colors.white, size: 20),
            ),
            const SizedBox(width: 8),
          ],
          Flexible(
            child: Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: message.isUser ? const Color(0xFF2E86AB) : Colors.grey[100],
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
            backgroundColor: const Color(0xFF2E86AB),
            child: const Icon(Icons.smart_toy, color: Colors.white, size: 20),
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
                hintText: 'Ask CysticCare AI about Polycystic Kidney Disease...',
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(25),
                  borderSide: BorderSide(color: Colors.grey[300]!),
                ),
                focusedBorder: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(25),
                  borderSide: const BorderSide(color: Color(0xFF2E86AB)),
                ),
                contentPadding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
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
          const DrawerHeader(
            decoration: BoxDecoration(
              color: Color(0xFF2E86AB),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Icon(
                  Icons.medical_services,
                  color: Colors.white,
                  size: 48,
                ),
                SizedBox(height: 16),
                Text(
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
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
              ),
            ),
          ),
          ...quickQuestions.map((question) => ListTile(
                title: Text(
                  question,
                  style: const TextStyle(fontSize: 14),
                ),
                onTap: () {
                  Navigator.pop(context);
                  _sendMessage(question);
                },
              )),
          const Divider(),
          const Padding(
            padding: EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  '⚠️ Disclaimer',
                  style: TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                SizedBox(height: 8),
                Text(
                  'This AI assistant provides general information only. Always consult healthcare professionals for medical advice.',
                  style: TextStyle(
                    fontSize: 12,
                    color: Colors.grey,
                  ),
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
