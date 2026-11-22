//! The tool abstraction: a name, a JSON input schema and an async handler.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde_json::{json, Value};

/// A boxed `Send` future, the return type of every tool handler.
pub type BoxFuture<T> = Pin<Box<dyn Future<Output = T> + Send + 'static>>;

type Handler = Arc<dyn Fn(Value) -> BoxFuture<Result<String, String>> + Send + Sync>;

/// One callable tool.
#[derive(Clone)]
pub struct Tool {
    /// Unique tool name.
    pub name: String,
    /// What the tool does, for the agent.
    pub description: String,
    /// JSON Schema of the `arguments` object.
    pub input_schema: Value,
    handler: Handler,
}

impl std::fmt::Debug for Tool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tool")
            .field("name", &self.name)
            .field("description", &self.description)
            .finish_non_exhaustive()
    }
}

impl Tool {
    /// A tool whose handler receives the parsed `arguments` (always a JSON
    /// object; `{}` when the caller sent none) and returns the text to show the
    /// agent, or an error text (reported to the agent as `isError: true`).
    pub fn new<F, Fut>(
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: Value,
        handler: F,
    ) -> Self
    where
        F: Fn(Value) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<String, String>> + Send + 'static,
    {
        Self {
            name: name.into(),
            description: description.into(),
            input_schema,
            handler: Arc::new(move |args| Box::pin(handler(args))),
        }
    }

    /// Run the handler.
    pub async fn call(&self, arguments: Value) -> Result<String, String> {
        (self.handler)(arguments).await
    }

    /// The `tools/list` entry for this tool.
    #[must_use]
    pub fn descriptor(&self) -> Value {
        json!({
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        })
    }
}

/// The set of tools a binary offers, in registration order.
#[derive(Debug, Clone, Default)]
pub struct ToolRegistry {
    tools: Vec<Tool>,
}

impl ToolRegistry {
    /// An empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a tool. A later tool with the same name replaces the earlier one.
    pub fn register(&mut self, tool: Tool) {
        self.tools.retain(|t| t.name != tool.name);
        self.tools.push(tool);
    }

    /// Builder form of [`ToolRegistry::register`].
    #[must_use]
    pub fn with(mut self, tool: Tool) -> Self {
        self.register(tool);
        self
    }

    /// Look a tool up by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&Tool> {
        self.tools.iter().find(|t| t.name == name)
    }

    /// Every tool's name, in registration order.
    #[must_use]
    pub fn names(&self) -> Vec<&str> {
        self.tools.iter().map(|t| t.name.as_str()).collect()
    }

    /// The `tools/list` array.
    #[must_use]
    pub fn descriptors(&self) -> Vec<Value> {
        self.tools.iter().map(Tool::descriptor).collect()
    }
}

/// Fetch a required string argument.
///
/// # Errors
///
/// A message naming the missing/ill-typed argument.
pub fn arg_str<'a>(args: &'a Value, key: &str) -> Result<&'a str, String> {
    args.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("argument `{key}` (string) is required"))
}

/// Fetch an optional finite number argument.
///
/// # Errors
///
/// A message when the argument is present but not a finite number.
pub fn arg_f64_opt(args: &Value, key: &str) -> Result<Option<f64>, String> {
    match args.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => v
            .as_f64()
            .filter(|n| n.is_finite())
            .map(Some)
            .ok_or_else(|| format!("argument `{key}` must be a finite number")),
    }
}

/// Fetch an optional unsigned integer argument.
///
/// # Errors
///
/// A message when the argument is present but not a non-negative integer.
pub fn arg_u64_opt(args: &Value, key: &str) -> Result<Option<u64>, String> {
    match args.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => v
            .as_u64()
            .map(Some)
            .ok_or_else(|| format!("argument `{key}` must be a non-negative integer")),
    }
}
