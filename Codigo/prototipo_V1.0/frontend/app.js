(function () {
	const COURSES = [
		{
			id: "fdp-123",
			name: "Fundamentos de Programaci\u00f3n - NRC 123",
			category: "Fundamentos de Programaci\u00f3n",
			cover: "cover-amber",
			progress: "35%",
			description: "Aula de prueba enfocada en bases de logica de programacion y resolucion de problemas paso a paso.",
			units: [
				"Unidad 1: Variables y tipos de datos",
				"Unidad 2: Condicionales",
				"Unidad 3: Ciclos y estructuras repetitivas"
			],
			resources: [
				"Syllabus.pdf",
				"Guia de ejercicios.pdf",
				"Material de apoyo.pdf"
			]
		},
		{
			id: "fdp-456",
			name: "Fundamentos de Programaci\u00f3n - NRC 456",
			category: "Fundamentos de Programaci\u00f3n",
			cover: "cover-violet",
			progress: "62%",
			description: "Aula de simulacion para practicar estructuras de control, algoritmos sencillos y trazado de ejecucion.",
			units: [
				"Unidad 1: Variables y tipos de datos",
				"Unidad 2: Condicionales",
				"Unidad 3: Ciclos y estructuras repetitivas"
			],
			resources: [
				"Syllabus.pdf",
				"Guia de ejercicios.pdf",
				"Material de apoyo.pdf"
			]
		},
		{
			id: "fdp-789",
			name: "Fundamentos de Programaci\u00f3n - NRC 789",
			category: "Fundamentos de Programaci\u00f3n",
			cover: "cover-teal",
			progress: "100%",
			description: "Aula de validacion del prototipo para reforzar fundamentos, ejercicios guiados y evaluaciones cortas.",
			units: [
				"Unidad 1: Variables y tipos de datos",
				"Unidad 2: Condicionales",
				"Unidad 3: Ciclos y estructuras repetitivas"
			],
			resources: [
				"Syllabus.pdf",
				"Guia de ejercicios.pdf",
				"Material de apoyo.pdf"
			]
		}
	];

	const state = {
		searchText: "",
		sortBy: "name-asc",
		currentCourseId: ""
	};

	const dom = {
		coursesView: document.getElementById("courses-view"),
		detailView: document.getElementById("course-detail-view"),
		coursesGrid: document.getElementById("courses-grid"),
		emptyState: document.getElementById("empty-state"),
		searchInput: document.getElementById("search-input"),
		sortSelect: document.getElementById("sort-select"),
		backButton: document.getElementById("back-button"),
		detailTitle: document.getElementById("detail-title"),
		detailCategory: document.getElementById("detail-category"),
		detailDescription: document.getElementById("detail-description"),
		detailUnits: document.getElementById("detail-units"),
		detailResources: document.getElementById("detail-resources")
	};

	function normalizeText(value) {
		return value
			.toLowerCase()
			.normalize("NFD")
			.replaceAll(/[\u0300-\u036f]/g, "");
	}

	function getVisibleCourses() {
		const query = normalizeText(state.searchText.trim());

		const filtered = COURSES.filter(function (course) {
			const searchable = normalizeText(course.name + " " + course.category);
			return searchable.includes(query);
		});

		filtered.sort(function (a, b) {
			if (state.sortBy === "name-desc") {
				return b.name.localeCompare(a.name, "es");
			}

			return a.name.localeCompare(b.name, "es");
		});

		return filtered;
	}

	function createCourseCard(course) {
		const card = document.createElement("article");
		let progressMarkup = "";

		if (course.progress) {
			progressMarkup = '<span class="progress-badge">' + course.progress + " completado</span>";
		}

		card.className = "course-card";
		card.dataset.courseId = course.id;
		card.setAttribute("tabindex", "0");
		card.setAttribute("role", "button");
		card.setAttribute("aria-label", "Abrir curso " + course.name);

		card.innerHTML =
			'<div class="course-cover ' + course.cover + '">' +
				progressMarkup +
			"</div>" +
			'<div class="course-body">' +
				'<h3 class="course-title">' + course.name + "</h3>" +
				'<p class="course-category">' + course.category + "</p>" +
				'<div class="course-actions">' +
					'<button type="button" class="course-enter-btn" data-course-id="' + course.id + '">Ingresar</button>' +
					'<button type="button" class="course-menu-btn" aria-label="Mas opciones">...</button>' +
				"</div>" +
			"</div>";

		return card;
	}

	function renderCourses() {
		const visibleCourses = getVisibleCourses();
		const fragment = document.createDocumentFragment();

		dom.coursesGrid.innerHTML = "";

		if (!visibleCourses.length) {
			dom.emptyState.classList.remove("hidden");
			return;
		}

		dom.emptyState.classList.add("hidden");

		visibleCourses.forEach(function (course) {
			fragment.appendChild(createCourseCard(course));
		});

		dom.coursesGrid.appendChild(fragment);
	}

	function renderCourseDetail(course) {
		dom.detailTitle.textContent = course.name;
		dom.detailCategory.textContent = course.category;
		dom.detailDescription.textContent = course.description;

		dom.detailUnits.innerHTML = "";
		course.units.forEach(function (unit) {
			const li = document.createElement("li");
			li.textContent = unit;
			dom.detailUnits.appendChild(li);
		});

		dom.detailResources.innerHTML = "";
		course.resources.forEach(function (resource) {
			const li = document.createElement("li");
			const link = document.createElement("a");

			link.className = "resource-link";
			link.href = "#";
			link.textContent = "Descargar " + resource;
			link.addEventListener("click", function (event) {
				event.preventDefault();
				globalThis.alert("Recurso simulado: " + resource);
			});

			li.appendChild(link);
			dom.detailResources.appendChild(li);
		});
	}

	function showCoursesView() {
		dom.detailView.classList.remove("view-active");
		dom.coursesView.classList.add("view-active");
	}

	function showDetailView() {
		dom.coursesView.classList.remove("view-active");
		dom.detailView.classList.add("view-active");
	}

	function openCourse(courseId) {
		const selected = COURSES.find(function (course) {
			return course.id === courseId;
		});

		if (!selected) {
			return;
		}

		state.currentCourseId = courseId;
		renderCourseDetail(selected);
		showDetailView();
		globalThis.scrollTo({ top: 0, behavior: "smooth" });
	}

	function onGridClick(event) {
		const enterButton = event.target.closest(".course-enter-btn");
		const menuButton = event.target.closest(".course-menu-btn");
		const card = event.target.closest(".course-card");

		if (enterButton) {
			openCourse(enterButton.dataset.courseId);
			return;
		}

		if (menuButton) {
			globalThis.alert("Menu del curso en desarrollo.");
			return;
		}

		if (card) {
			openCourse(card.dataset.courseId);
		}
	}

	function onGridKeydown(event) {
		const card = event.target.closest(".course-card");

		if (!card) {
			return;
		}

		if (event.key === "Enter" || event.key === " ") {
			event.preventDefault();
			openCourse(card.dataset.courseId);
		}
	}

	function bindEvents() {
		dom.searchInput.addEventListener("input", function (event) {
			state.searchText = event.target.value;
			renderCourses();
		});

		dom.sortSelect.addEventListener("change", function (event) {
			state.sortBy = event.target.value;
			renderCourses();
		});

		dom.coursesGrid.addEventListener("click", onGridClick);
		dom.coursesGrid.addEventListener("keydown", onGridKeydown);

		dom.backButton.addEventListener("click", function () {
			showCoursesView();
			state.currentCourseId = "";
		});
	}

	function init() {
		renderCourses();
		bindEvents();
	}

	// El flujo queda listo para conectar APIs reales en futuras iteraciones.
	document.addEventListener("DOMContentLoaded", init);
})();
