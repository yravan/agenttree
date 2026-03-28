"""Browser action tool schemas for LLM tool calling."""

BROWSER_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "navigate",
            "description": "Navigate to a URL in the current tab",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to navigate to"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "click",
            "description": "Click an element by its reference ID from the DOM serialization",
            "parameters": {
                "type": "object",
                "properties": {
                    "ref": {"type": "string", "description": "Element reference ID (e.g., 'e1', 'e2')"},
                },
                "required": ["ref"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "type_text",
            "description": "Type text into an input element",
            "parameters": {
                "type": "object",
                "properties": {
                    "ref": {"type": "string", "description": "Element reference ID"},
                    "text": {"type": "string", "description": "Text to type"},
                    "clear_first": {
                        "type": "boolean",
                        "description": "Clear existing text before typing",
                        "default": True,
                    },
                },
                "required": ["ref", "text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scroll",
            "description": "Scroll the page",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["up", "down"],
                        "description": "Scroll direction",
                    },
                    "amount": {
                        "type": "integer",
                        "description": "Number of pixels to scroll",
                        "default": 500,
                    },
                },
                "required": ["direction"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "select",
            "description": "Select an option from a dropdown",
            "parameters": {
                "type": "object",
                "properties": {
                    "ref": {"type": "string", "description": "Element reference ID"},
                    "value": {"type": "string", "description": "Option value to select"},
                },
                "required": ["ref", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wait",
            "description": "Wait for a specified number of seconds",
            "parameters": {
                "type": "object",
                "properties": {
                    "seconds": {
                        "type": "number",
                        "description": "Seconds to wait",
                        "default": 2,
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract",
            "description": "Extract specific data from the current page",
            "parameters": {
                "type": "object",
                "properties": {
                    "goal": {
                        "type": "string",
                        "description": "What data to extract from the page",
                    },
                },
                "required": ["goal"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "done",
            "description": "Mark the task as complete and return the result",
            "parameters": {
                "type": "object",
                "properties": {
                    "result": {
                        "type": "string",
                        "description": "The extracted data or task result (JSON string preferred)",
                    },
                    "success": {
                        "type": "boolean",
                        "description": "Whether the task was completed successfully",
                        "default": True,
                    },
                },
                "required": ["result"],
            },
        },
    },
]
